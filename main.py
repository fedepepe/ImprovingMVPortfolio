import pandas as pd
import numpy as np
import math
import scipy.optimize as opt
import matplotlib.pyplot as plt
import proposed_estim
import label_lines


def quadratic_util_fun(w, mu, cov_mat, phi):
    # w  : vector of weights
    # mu : vector of mean returns
    # cov_mat : covariance matrix of returns
    # phi : risk aversion coefficient

    if phi is None:
        # Compute risk aversion factor yielding the tangency portfolio
        # Sigma_np = cov_mat.to_numpy()
        # invSigma_np = np.linalg.inv(Sigma_np)
        # invSigma = pd.DataFrame(data = invSigma_np, index = cov_mat.index, columns = cov_mat.columns)
        # series_of_ones = pd.Series(1, index=mu.index)
        # phi0 = series_of_ones.dot(invSigma).dot(mu)
        # return phi0/2 * w.dot(cov_mat).dot(w) - w.dot(mu)  # Tangency portfolio
        # Alternative single-line (equivalent) code:
        return - w.dot(mu) / math.sqrt(abs(w.dot(cov_mat).dot(w)))  # Tangency portfolio
        # abs just prevents the run from failing due to numerical errors (causing w*Sigma*w being < 0)
    elif math.isinf(phi):
        return w.dot(cov_mat).dot(w)  # Minimum variance portfolio
    else:
        return phi / 2. * w.dot(cov_mat).dot(w) - w.dot(mu)  # Optimized portfolio


def leverage(w):
    return abs(w).sum()


def plot_portfolio(mu_p, sig2_p, sym_str='bo', figure=None):
    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
    else:
        ax = figure.axes[0]
    ax.plot(100. * np.sqrt(252 * sig2_p), 100. * 252 * mu_p, sym_str, markersize=3)
    return figure


def main():
    # Load dataset of daily returns and volatilities
    [_, return_df, real_vol_df, mktcap_df, _] = pd.read_pickle('SP500_thesis.pkl')

    # Remove columns with at least one NaN
    return_df = return_df[return_df.columns.intersection(mktcap_df.columns)].dropna(axis=1)
    real_vol_df = real_vol_df[real_vol_df.columns.intersection(mktcap_df.columns)].dropna(axis=1)

    # %% Set simulation parameters
    allow_short_sell = False
    risk_avers = np.inf  # Risk aversion coefficient for quadratic utility function
    max_lvrg = 3.  # Maximum leverage (i.e., L1 norm of portfolio weights vector)
    max_weight = .5  # Maximum single portfolio weight (to promote diversification)

    n_stck = 150  # Size of investment universe
    n_run = 10  # Number of simulation runs

    # %% Do computations
    data = np.empty((0, 6), float)
    figure = None

    n_done = 0
    while n_done <= n_run:
        # Randomly pick n_stck stocks out of the SP500 components to form the investment universe
        idx_stk_sel = np.random.choice(len(return_df.columns), size=n_stck, replace=False)

        ret_df = return_df.iloc[:, idx_stk_sel]
        vol_df = real_vol_df.iloc[:, idx_stk_sel]

        stock_idx = ret_df.columns

        ret_df_train, ret_df_test = np.split(ret_df, [int(.75 * len(ret_df))])
        vol_df_train, vol_df_test = np.split(vol_df, [int(.75 * len(vol_df))])

        series_of_ones = pd.Series(1, index=stock_idx)

        # Imposing sum of weights being equal to 1
        linear_constraint = opt.LinearConstraint(series_of_ones, 1, 1)

        # Imposing the restricion on maximum leverage
        nonlinear_constraint = opt.NonlinearConstraint(leverage, 1, max_lvrg)

        # Imposing the no short-selling restriction
        if allow_short_sell:
            bounds = opt.Bounds(-max_weight * series_of_ones, max_weight * series_of_ones)
        else:
            bounds = opt.Bounds(0 * series_of_ones, max_weight * series_of_ones)

        # Equally-weighted portfolio
        w_eq = 1. / n_stck * series_of_ones

        # Initial starting point is either the equally-weighted portfolio or a random one
        # w0 = w_eq
        w0 = np.random.uniform(low=0.0, high=1.0, size=n_stck) * series_of_ones
        w0 = w0 / sum(w0)

        ### Optimize portfolio variance over the training set
        # 1. Sigma is the sample covmat of returns
        mu = ret_df_train.mean().values  # vector of mean returns
        cov_mat = ret_df_train.cov().values  # covariance matrix

        opt_result = opt.minimize(quadratic_util_fun, w0, args=(mu, cov_mat, risk_avers),
                                  method='trust-constr',
                                  options={'verbose': False, 'maxiter': 250},
                                  constraints=(linear_constraint, nonlinear_constraint),
                                  bounds=bounds)

        # 2. Sigma is given by the proposed estimator
        mu_impr, cov_mat_impr = proposed_estim.proposed_estim(ret_df_train, vol_df_train)

        opt_result_impr = opt.minimize(quadratic_util_fun, w0, args=(mu_impr, cov_mat_impr, risk_avers),
                                       method='trust-constr',
                                       options={'verbose': False, 'maxiter': 250},
                                       constraints=(linear_constraint, nonlinear_constraint),
                                       bounds=bounds)

        ### Compute out-of-sample mean and variance and plot
        if opt_result.success and opt_result_impr.success:
            # Using sample covariance matrix of returns
            w_opt = pd.Series(opt_result.x, index=stock_idx)
            pf_ret = ret_df_test.values.dot(w_opt)
            mu_p = pf_ret.mean()
            sig2_p = pf_ret.var()
            figure = plot_portfolio(mu_p, sig2_p, 'sb', figure)

            # Using proposed estimator of covariance matrix
            w_opt_impr = pd.Series(opt_result_impr.x, index=stock_idx)
            pf_ret = ret_df_test.values.dot(w_opt_impr)
            mu_p_impr = pf_ret.mean()
            sig2_p_impr = pf_ret.var()
            figure = plot_portfolio(mu_p_impr, sig2_p_impr, 'dr', figure)

            # Using identity matrix (i.e., equally-weighted portfolio)
            w_opt_ew = w_eq
            pf_ret = ret_df_test.values.dot(w_opt_ew)
            mu_p_ew = pf_ret.mean()
            sig2_p_ew = pf_ret.var()
            figure = plot_portfolio(mu_p_ew, sig2_p_ew, 'y*', figure)

            # Print data to text file
            new_data = np.array([[mu_p, sig2_p, mu_p_impr, sig2_p_impr, mu_p_ew, sig2_p_ew]])
            data = np.append(data, new_data, axis=0)

            with open(f'pf_mu_sig_stk{n_stck}_rac{risk_avers}_lvg{max_lvrg}.txt', "a") as f:
                np.savetxt(f, new_data, fmt='%.8f')

            n_done += 1

            print(f'{n_done}', end='..')

    ax = figure.axes[0]
    ax.set_xlabel('Standard deviation [%]', fontsize=12)
    ax.set_ylabel('Return [%]', fontsize=12)
    ax.legend(['Sample cov. matrix', 'Proposed estim.', 'Equally weighted'])

    # %% Plot contour lines with equal Sharpe ratio
    xlim = np.asarray(ax.get_xlim())
    plt.autoscale(False)  # Turn autoscaling off
    SR = [0.25, 0.5, 0.75, 1.]
    for sr in SR:
        ax.plot(xlim, sr * xlim, '--', linewidth=1, label='SR=' + str(sr))

    label_lines.labelLines(ax.get_lines()[-len(SR):], zorder=2.5, fontsize=8)
    plt.show()


if __name__ == '__main__':
    main()

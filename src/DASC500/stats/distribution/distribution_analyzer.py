import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma, beta as beta_func
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Callable
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import itertools

# Import dedicated plotting functions
from .plotting import (
    plot_transformations,
)
from .plotting import plot_synthetic_data
from .plotting import plot_mixture_model
from .plotting import (
    plot_bayesian_inference,
)
from .plotting import (
    plot_distribution_distance,
)
from .plotting import (
    plot_parameter_optimization,
)
from .plotting import (
    plot_distribution_comparison,
)
from .plotting import (
    plot_goodness_of_fit,
)
from .plotting import (
    plot_confidence_intervals,
)
from .plotting import (
    plot_probability_plot,
)

warnings.filterwarnings("ignore")


class DistributionAnalyzer:
    """
    A comprehensive toolkit for analyzing and working with statistical distributions.
    Supports: Normal, Student's t, Chi-squared, F, Lognormal, Exponential, Beta, Uniform,
    Bernoulli, Geometric, Binomial, and Poisson distributions with advanced capabilities.
    """

    def __init__(self):
        self.distributions = {
            "normal": stats.norm,
            "student_t": stats.t,
            "chi_squared": stats.chi2,
            "f": stats.f,
            "lognormal": stats.lognorm,
            "exponential": stats.expon,
            "beta": stats.beta,
            "uniform": stats.uniform,
            "bernoulli": stats.bernoulli,
            "geometric": stats.geom,
            "binomial": stats.binom,
            "poisson": stats.poisson,
            "gamma": stats.gamma,
            "weibull": stats.weibull_min,
            "pareto": stats.pareto,
            "laplace": stats.laplace,
        }

        self.continuous_distributions = [
            "normal",
            "student_t",
            "chi_squared",
            "f",
            "lognormal",
            "exponential",
            "beta",
            "uniform",
            "gamma",
            "weibull",
            "pareto",
            "laplace",
        ]

        self.discrete_distributions = ["bernoulli", "geometric", "binomial", "poisson"]

    def generate_samples(
        self, distribution: str, size: int = 1000, **params
    ) -> np.ndarray:
        """Enhanced sample generation with parameter validation."""
        if distribution not in self.distributions:
            raise ValueError(f"Distribution '{distribution}' not supported")

        dist = self.distributions[distribution]

        # Expanded default parameters
        default_params = {
            "normal": {"loc": 0, "scale": 1},
            "student_t": {"df": 5, "loc": 0, "scale": 1},
            "chi_squared": {"df": 5},
            "f": {"dfn": 5, "dfd": 10},
            "lognormal": {"s": 1, "loc": 0, "scale": 1},
            "exponential": {"loc": 0, "scale": 1},
            "beta": {"a": 2, "b": 2, "loc": 0, "scale": 1},
            "uniform": {"loc": 0, "scale": 1},
            "bernoulli": {"p": 0.5},
            "geometric": {"p": 0.3},
            "binomial": {"n": 10, "p": 0.5},
            "poisson": {"mu": 3},
            "gamma": {"a": 2, "loc": 0, "scale": 1},
            "weibull": {"c": 1.5, "loc": 0, "scale": 1},
            "pareto": {"b": 1, "loc": 0, "scale": 1},
            "laplace": {"loc": 0, "scale": 1},
        }

        final_params = default_params.get(distribution, {})
        final_params.update(params)

        return dist.rvs(size=size, **final_params)

    def calculate_statistics(
        self, data: np.ndarray, include_advanced: bool = True
    ) -> Dict[str, float]:
        """Calculate comprehensive statistics including advanced measures."""
        basic_stats = {
            "count": len(data),
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data, ddof=1),
            "variance": np.var(data, ddof=1),
            "min": np.min(data),
            "max": np.max(data),
            "range": np.max(data) - np.min(data),
            "q25": np.percentile(data, 25),
            "q50": np.percentile(data, 50),
            "q75": np.percentile(data, 75),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25),
        }

        if include_advanced:
            try:
                mode_result = stats.mode(data, keepdims=True)
                mode_value = (
                    mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan
                )
            except:
                mode_value = np.nan

            advanced_stats = {
                "mode": mode_value,
                "skewness": stats.skew(data),
                "kurtosis": stats.kurtosis(data),
                "excess_kurtosis": stats.kurtosis(data, fisher=True),
                "coefficient_of_variation": (
                    np.std(data) / np.mean(data) if np.mean(data) != 0 else np.nan
                ),
                "mad": np.median(
                    np.abs(data - np.median(data))
                ),  # Median Absolute Deviation
                "geometric_mean": (
                    stats.gmean(data[data > 0]) if np.all(data > 0) else np.nan
                ),
                "harmonic_mean": (
                    stats.hmean(data[data > 0]) if np.all(data > 0) else np.nan
                ),
                "trimmed_mean_10": stats.trim_mean(data, 0.1),
                "trimmed_mean_25": stats.trim_mean(data, 0.25),
                "entropy": stats.entropy(np.histogram(data, bins=30)[0] + 1e-10),
                "moment_2": stats.moment(data, 2),
                "moment_3": stats.moment(data, 3),
                "moment_4": stats.moment(data, 4),
            }
            basic_stats.update(advanced_stats)

        return basic_stats

    def fit_distribution(
        self, data: np.ndarray, distribution: str, method: str = "mle"
    ) -> Tuple[tuple, Dict[str, float]]:
        """Enhanced distribution fitting with multiple methods and detailed results."""
        if distribution not in self.distributions:
            raise ValueError(f"Distribution '{distribution}' not supported")

        dist = self.distributions[distribution]

        # Fit the distribution
        if method == "mle":
            params = dist.fit(data)
        elif method == "moments":
            params = self._fit_by_moments(data, distribution)
        else:
            raise ValueError(f"Method '{method}' not supported")

        # Calculate multiple goodness of fit measures
        gof_results = self._calculate_goodness_of_fit(data, distribution, params)

        return params, gof_results

    def _fit_by_moments(self, data: np.ndarray, distribution: str) -> tuple:
        """Fit distribution using method of moments."""
        mean = np.mean(data)
        var = np.var(data, ddof=1)

        if distribution == "normal":
            return (mean, np.sqrt(var))
        elif distribution == "exponential":
            return (0, mean)
        elif distribution == "gamma":
            # Method of moments for gamma distribution
            shape = mean**2 / var
            scale = var / mean
            return (shape, 0, scale)
        elif distribution == "beta":
            # Method of moments for beta distribution (assuming loc=0, scale=1)
            if mean <= 0 or mean >= 1:
                raise ValueError("Beta distribution requires data in (0,1)")
            a = mean * (mean * (1 - mean) / var - 1)
            b = (1 - mean) * (mean * (1 - mean) / var - 1)
            return (a, b, 0, 1)
        else:
            # Fall back to MLE for other distributions
            return self.distributions[distribution].fit(data)

    def _calculate_goodness_of_fit(
        self, data: np.ndarray, distribution: str, params: tuple
    ) -> Dict[str, float]:
        """Calculate comprehensive goodness of fit statistics."""
        dist = self.distributions[distribution]

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, lambda x: dist.cdf(x, *params))

        # Anderson-Darling test (for continuous distributions)
        ad_stat, ad_p = np.nan, np.nan
        if distribution in self.continuous_distributions:
            try:
                ad_result = stats.anderson(data, dist="norm")
                ad_stat = ad_result.statistic
                # Convert to p-value approximation
                ad_p = 1 - stats.norm.cdf(ad_stat)
            except:
                pass

        # Log-likelihood and information criteria
        try:
            if distribution in self.continuous_distributions:
                log_likelihood = np.sum(dist.logpdf(data, *params))
            else:
                log_likelihood = np.sum(dist.logpmf(data, *params))

            k = len(params)  # number of parameters
            n = len(data)  # sample size

            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else np.inf
        except:
            log_likelihood = aic = bic = aicc = np.nan

        # Root Mean Square Error for fitted vs empirical CDF
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fitted_cdf = dist.cdf(sorted_data, *params)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf) ** 2))

        return {
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic,
            "aicc": aicc,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
            "ad_statistic": ad_stat,
            "ad_p_value": ad_p,
            "rmse_cdf": rmse,
        }

    def compare_distributions(
        self, data: np.ndarray, distributions: List[str] = None, criteria: str = "aic"
    ) -> pd.DataFrame:
        """Enhanced distribution comparison with multiple criteria."""
        if distributions is None:
            if np.all(data >= 0):
                distributions = [
                    "normal",
                    "lognormal",
                    "exponential",
                    "gamma",
                    "weibull",
                ]
            else:
                distributions = ["normal", "laplace", "student_t"]

        results = []

        for dist_name in distributions:
            try:
                params, gof = self.fit_distribution(data, dist_name)

                result = {
                    "distribution": dist_name,
                    "parameters": params,
                    "num_params": len(params),
                }
                result.update(gof)
                results.append(result)

            except Exception as e:
                print(f"Could not fit {dist_name}: {e}")
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            # Sort by specified criteria (lower is better for most)
            if criteria in ["aic", "bic", "aicc", "rmse_cdf"]:
                df = df.sort_values(criteria, ascending=True)
            elif criteria in ["ks_p_value", "ad_p_value"]:
                df = df.sort_values(criteria, ascending=False)
            else:
                df = df.sort_values("aic", ascending=True)

        return df

    def distribution_properties(
        self, distribution: str, **params
    ) -> Dict[str, Union[float, str]]:
        """Get theoretical properties of a distribution."""
        if distribution not in self.distributions:
            raise ValueError(f"Distribution '{distribution}' not supported")

        dist = self.distributions[distribution]

        # Use default parameters if none provided
        default_params = {
            "normal": {"loc": 0, "scale": 1},
            "student_t": {"df": 5},
            "chi_squared": {"df": 5},
            "f": {"dfn": 5, "dfd": 10},
            "lognormal": {"s": 1},
            "exponential": {"scale": 1},
            "beta": {"a": 2, "b": 2},
            "uniform": {"loc": 0, "scale": 1},
            "bernoulli": {"p": 0.5},
            "geometric": {"p": 0.3},
            "binomial": {"n": 10, "p": 0.5},
            "poisson": {"mu": 3},
            "gamma": {"a": 2},
            "weibull": {"c": 1.5},
            "pareto": {"b": 1},
            "laplace": {"loc": 0, "scale": 1},
        }

        final_params = default_params.get(distribution, {})
        final_params.update(params)

        try:
            properties = {
                "mean": dist.mean(**final_params),
                "variance": dist.var(**final_params),
                "std": dist.std(**final_params),
                "skewness": dist.stats(moments="s", **final_params),
                "kurtosis": dist.stats(moments="k", **final_params),
                "support": self._get_support(distribution, **final_params),
                "type": (
                    "continuous"
                    if distribution in self.continuous_distributions
                    else "discrete"
                ),
            }

            # Add distribution-specific properties
            if distribution == "normal":
                properties["68_95_99_rule"] = (
                    "68% within 1σ, 95% within 2σ, 99.7% within 3σ"
                )
            elif distribution == "exponential":
                properties["memoryless"] = True
                properties["rate"] = 1 / final_params["scale"]
            elif distribution == "poisson":
                properties["rate"] = final_params["mu"]
                properties["variance_equals_mean"] = True
            elif distribution == "binomial":
                properties["success_probability"] = final_params["p"]
                properties["num_trials"] = final_params["n"]
            elif distribution == "geometric":
                properties["success_probability"] = final_params["p"]
                properties["memoryless"] = True

        except Exception as e:
            properties = {"error": str(e)}

        return properties

    def _get_support(self, distribution: str, **params) -> str:
        """Get the support (domain) of a distribution."""
        support_map = {
            "normal": "(-∞, ∞)",
            "student_t": "(-∞, ∞)",
            "chi_squared": "[0, ∞)",
            "f": "[0, ∞)",
            "lognormal": "[0, ∞)",
            "exponential": "[0, ∞)",
            "beta": "[0, 1]",
            "uniform": f"[{params.get('loc', 0)}, {params.get('loc', 0) + params.get('scale', 1)}]",
            "bernoulli": "{0, 1}",
            "geometric": "{1, 2, 3, ...}",
            "binomial": f"{{0, 1, 2, ..., {params.get('n', 10)}}}",
            "poisson": "{0, 1, 2, ...}",
            "gamma": "[0, ∞)",
            "weibull": "[0, ∞)",
            "pareto": "[1, ∞)",
            "laplace": "(-∞, ∞)",
        }
        return support_map.get(distribution, "Unknown")

    def plot_distribution(
        self,
        data: np.ndarray = None,
        distribution: str = None,
        params: tuple = None,
        bins: int = 50,
        figsize: Tuple[int, int] = (15, 12),
        plot_type: str = "comprehensive",
    ):
        """Enhanced plotting with multiple visualization options."""

        if plot_type == "comprehensive":
            fig, axes = plt.subplots(3, 3, figsize=figsize)
            axes = axes.flatten()

            if data is not None:
                # Plot 1: Histogram with fitted distribution
                axes[0].hist(
                    data,
                    bins=bins,
                    density=True,
                    alpha=0.7,
                    color="skyblue",
                    edgecolor="black",
                    label="Data",
                )

                if distribution and distribution in self.distributions:
                    if params is None:
                        params, _ = self.fit_distribution(data, distribution)

                    dist = self.distributions[distribution]
                    x = np.linspace(np.min(data), np.max(data), 100)
                    if distribution in self.continuous_distributions:
                        axes[0].plot(
                            x,
                            dist.pdf(x, *params),
                            "r-",
                            linewidth=2,
                            label=f"Fitted {distribution}",
                        )

                axes[0].set_title("Histogram with Fitted Distribution")
                axes[0].set_xlabel("Value")
                axes[0].set_ylabel("Density")
                axes[0].legend()

                # Plot 2: Q-Q plot
                if distribution and distribution in self.distributions:
                    if params is None:
                        params, _ = self.fit_distribution(data, distribution)

                    dist = self.distributions[distribution]
                    stats.probplot(data, dist=dist, sparams=params, plot=axes[1])
                    axes[1].set_title(f"Q-Q Plot vs {distribution}")
                else:
                    stats.probplot(data, dist="norm", plot=axes[1])
                    axes[1].set_title("Q-Q Plot vs Normal")

                # Plot 3: Box plot
                axes[2].boxplot(data, vert=True)
                axes[2].set_title("Box Plot")
                axes[2].set_ylabel("Value")

                # Plot 4: Empirical vs Theoretical CDF
                sorted_data = np.sort(data)
                empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                axes[3].plot(
                    sorted_data, empirical_cdf, "b-", linewidth=2, label="Empirical CDF"
                )

                if distribution and distribution in self.distributions:
                    if params is None:
                        params, _ = self.fit_distribution(data, distribution)

                    dist = self.distributions[distribution]
                    theoretical_cdf = dist.cdf(sorted_data, *params)
                    axes[3].plot(
                        sorted_data,
                        theoretical_cdf,
                        "r-",
                        linewidth=2,
                        label=f"Theoretical {distribution} CDF",
                    )
                    axes[3].legend()

                axes[3].set_title("Empirical vs Theoretical CDF")
                axes[3].set_xlabel("Value")
                axes[3].set_ylabel("Probability")

                # Plot 5: Residuals (Empirical - Theoretical CDF)
                if distribution and distribution in self.distributions:
                    if params is None:
                        params, _ = self.fit_distribution(data, distribution)

                    dist = self.distributions[distribution]
                    residuals = empirical_cdf - dist.cdf(sorted_data, *params)
                    axes[4].plot(sorted_data, residuals, "g-", linewidth=2)
                    axes[4].axhline(y=0, color="r", linestyle="--", alpha=0.5)
                    axes[4].set_title("CDF Residuals")
                    axes[4].set_xlabel("Value")
                    axes[4].set_ylabel("Empirical - Theoretical")

                # Plot 6: Kernel Density Estimate
                try:
                    kde = stats.gaussian_kde(data)
                    x_kde = np.linspace(np.min(data), np.max(data), 200)
                    axes[5].plot(x_kde, kde(x_kde), "purple", linewidth=2, label="KDE")
                    axes[5].hist(
                        data, bins=bins, density=True, alpha=0.3, color="lightgray"
                    )
                    axes[5].set_title("Kernel Density Estimate")
                    axes[5].set_xlabel("Value")
                    axes[5].set_ylabel("Density")
                    axes[5].legend()
                except:
                    axes[5].text(
                        0.5,
                        0.5,
                        "KDE not available",
                        ha="center",
                        va="center",
                        transform=axes[5].transAxes,
                    )

                # Plot 7: Autocorrelation
                try:
                    from statsmodels.tsa.stattools import acf

                    autocorr = acf(data, nlags=min(40, len(data) // 4))
                    axes[6].plot(autocorr, "o-", linewidth=2, markersize=4)
                    axes[6].axhline(y=0, color="r", linestyle="--", alpha=0.5)
                    axes[6].set_title("Autocorrelation Function")
                    axes[6].set_xlabel("Lag")
                    axes[6].set_ylabel("Autocorrelation")
                except:
                    # Fallback simple autocorrelation
                    lags = range(1, min(20, len(data) // 4))
                    autocorr = [
                        np.corrcoef(data[:-lag], data[lag:])[0, 1] for lag in lags
                    ]
                    axes[6].plot(lags, autocorr, "o-", linewidth=2, markersize=4)
                    axes[6].axhline(y=0, color="r", linestyle="--", alpha=0.5)
                    axes[6].set_title("Autocorrelation Function (Simple)")
                    axes[6].set_xlabel("Lag")
                    axes[6].set_ylabel("Autocorrelation")

                # Plot 8: Distribution of standardized residuals
                if distribution and distribution in self.distributions:
                    if params is None:
                        params, _ = self.fit_distribution(data, distribution)

                    dist = self.distributions[distribution]
                    if distribution in self.continuous_distributions:
                        # Transform to uniform [0,1] then to standard normal
                        uniform_data = dist.cdf(data, *params)
                        std_residuals = stats.norm.ppf(uniform_data)
                        std_residuals = std_residuals[np.isfinite(std_residuals)]

                        if len(std_residuals) > 0:
                            axes[7].hist(
                                std_residuals,
                                bins=30,
                                density=True,
                                alpha=0.7,
                                color="lightcoral",
                                edgecolor="black",
                            )
                            x_norm = np.linspace(
                                np.min(std_residuals), np.max(std_residuals), 100
                            )
                            axes[7].plot(
                                x_norm,
                                stats.norm.pdf(x_norm),
                                "k-",
                                linewidth=2,
                                label="Standard Normal",
                            )
                            axes[7].set_title("Standardized Residuals")
                            axes[7].set_xlabel("Standardized Residual")
                            axes[7].set_ylabel("Density")
                            axes[7].legend()

                # Plot 9: Summary statistics visualization
                stats_dict = self.calculate_statistics(data)
                key_stats = ["mean", "median", "std", "skewness", "kurtosis"]
                values = [stats_dict.get(stat, 0) for stat in key_stats]

                axes[8].bar(
                    key_stats, values, color="lightsteelblue", edgecolor="black"
                )
                axes[8].set_title("Key Statistics")
                axes[8].set_ylabel("Value")
                axes[8].tick_params(axis="x", rotation=45)

                # Add value labels on bars
                for i, v in enumerate(values):
                    axes[8].text(
                        i,
                        v + 0.01 * max(abs(max(values)), abs(min(values))),
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

            else:
                # Plot theoretical distribution properties
                if distribution and distribution in self.distributions:
                    dist = self.distributions[distribution]
                    if params is None:
                        # Use default parameters
                        default_params = {
                            "normal": (0, 1),
                            "student_t": (5,),
                            "chi_squared": (5,),
                            "exponential": (1,),
                            "beta": (2, 2),
                            "uniform": (0, 1),
                            "poisson": (3,),
                            "gamma": (2,),
                            "weibull": (1.5,),
                            "laplace": (0, 1),
                        }
                        params = default_params.get(distribution, (1,))

                    if distribution in self.continuous_distributions:
                        # PDF plot
                        x = np.linspace(
                            dist.ppf(0.01, *params), dist.ppf(0.99, *params), 100
                        )
                        axes[0].plot(x, dist.pdf(x, *params), "b-", linewidth=2)
                        axes[0].set_title(f"{distribution.title()} PDF")
                        axes[0].set_xlabel("x")
                        axes[0].set_ylabel("Probability Density")

                        # CDF plot
                        axes[1].plot(x, dist.cdf(x, *params), "r-", linewidth=2)
                        axes[1].set_title(f"{distribution.title()} CDF")
                        axes[1].set_xlabel("x")
                        axes[1].set_ylabel("Cumulative Probability")
                    else:
                        # PMF plot for discrete distributions
                        x = np.arange(0, 20)
                        pmf_vals = dist.pmf(x, *params)
                        axes[0].bar(x, pmf_vals, alpha=0.7, color="blue")
                        axes[0].set_title(f"{distribution.title()} PMF")
                        axes[0].set_xlabel("x")
                        axes[0].set_ylabel("Probability")

                        # CDF plot
                        axes[1].step(x, dist.cdf(x, *params), where="post", linewidth=2)
                        axes[1].set_title(f"{distribution.title()} CDF")
                        axes[1].set_xlabel("x")
                        axes[1].set_ylabel("Cumulative Probability")

            plt.tight_layout()
            plt.show()

        elif plot_type == "comparison":
            # Plot for comparing multiple distributions
            self._plot_distribution_comparison(data, distribution, params, figsize)

    def plot_power_vs_alpha(
        self,
        effect_size: float,
        n: int,
        test_type: str,
        alpha_range: np.ndarray = np.linspace(0.0, 0.25, 100),
    ):
        """
        Plots how power changes with the significance level (alpha).
        """
        powers = []
        for alpha in alpha_range:
            # Use the power_analysis method to calculate power for each alpha
            power_result = self.power_analysis(
                effect_size=effect_size, n=n, alpha=alpha, test_type=test_type
            )
            powers.append(power_result["power"])

        plt.figure(figsize=(8, 6))
        plt.plot(alpha_range, powers, marker=".", linestyle="-")
        plt.title(
            f"Power vs. Alpha for {test_type}\n(Effect Size={effect_size:.2f}, n={n})"
        )
        plt.xlabel("Significance Level (Alpha)")
        plt.ylabel("Power (1 - β)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(0, 1.05)
        plt.show()

    def _plot_distribution_comparison(
        self,
        data: np.ndarray,
        distributions: List[str],
        params_list: List[tuple],
        figsize: Tuple[int, int],
    ):
        """Plot comparison of multiple distributions."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

        # Histogram with all fitted distributions
        axes[0].hist(
            data,
            bins=50,
            density=True,
            alpha=0.5,
            color="lightgray",
            edgecolor="black",
            label="Data",
        )

        x = np.linspace(np.min(data), np.max(data), 200)
        for i, (dist_name, params) in enumerate(zip(distributions, params_list)):
            if dist_name in self.distributions:
                dist = self.distributions[dist_name]
                if dist_name in self.continuous_distributions:
                    axes[0].plot(
                        x,
                        dist.pdf(x, *params),
                        color=colors[i % len(colors)],
                        linewidth=2,
                        label=f"{dist_name}",
                    )

        axes[0].set_title("Distribution Comparison")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # CDF comparison
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1].plot(sorted_data, empirical_cdf, "k-", linewidth=2, label="Empirical")

        for i, (dist_name, params) in enumerate(zip(distributions, params_list)):
            if dist_name in self.distributions:
                dist = self.distributions[dist_name]
                theoretical_cdf = dist.cdf(sorted_data, *params)
                axes[1].plot(
                    sorted_data,
                    theoretical_cdf,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    label=f"{dist_name}",
                )

        axes[1].set_title("CDF Comparison")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Cumulative Probability")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_transformations(self, data: np.ndarray, **kwargs):
        """Plot histograms and QQ plots for original and transformed data."""
        # Get transformation results
        transformation_results = self.optimal_transformation(data)

        # Use dedicated plotting function
        return plot_transformations(data, transformation_results, **kwargs)

    def plot_synthetic_data(
        self,
        distribution: str,
        size: int = 1000,
        with_noise: bool = False,
        noise_level: float = 0.1,
        **kwargs,
    ):
        """Plot synthetic data with histogram, theoretical PDF, and optional QQ plot and ECDF."""
        # Generate the synthetic data
        params = kwargs.pop("params", {})
        data = self.generate_synthetic_data(
            distribution, size, with_noise, noise_level, **params
        )

        # Use dedicated plotting function
        return plot_synthetic_data(
            data, distribution, params, with_noise, noise_level, **kwargs
        )

    def plot_mixture_model(
        self,
        data: np.ndarray,
        n_components: int = 2,
        distribution: str = "normal",
        **kwargs,
    ):
        """Plot mixture model with histogram and component PDFs."""
        # Fit mixture model
        mixture_results = self.fit_mixture_model(data, n_components, distribution)

        if "error" in mixture_results:
            print(f"Error fitting mixture model: {mixture_results['error']}")
            return

        # Use dedicated plotting function
        return plot_mixture_model(
            data, mixture_results, [distribution] * n_components, **kwargs
        )

    def plot_bayesian_inference(
        self,
        data: np.ndarray,
        distribution: str,
        prior_params: Dict = None,
        n_samples: int = 1000,
        **kwargs,
    ):
        """Plot Bayesian inference results for distribution parameters."""
        # Perform Bayesian inference
        inference_results = self.bayesian_inference(
            data, distribution, prior_params, n_samples
        )

        if "error" in inference_results:
            print(f"Error in Bayesian inference: {inference_results['error']}")
            return

        # Use dedicated plotting function
        return plot_bayesian_inference(data, inference_results, **kwargs)

    def plot_distribution_distance(
        self, data1: np.ndarray, data2: np.ndarray, method: str = "ks", **kwargs
    ):
        """Plot distribution distance comparison with histograms and optional ECDF."""
        # Calculate distribution distance
        distance_results = self.distribution_distance(data1, data2, method)

        # Use dedicated plotting function
        return plot_distribution_distance(data1, data2, distance_results, **kwargs)

    def plot_parameter_optimization(
        self,
        distribution: str,
        data: np.ndarray,
        method: str = "mle",
        bounds: Dict = None,
        **kwargs,
    ):
        """Plot parameter optimization results with histogram, fitted PDF, and optional QQ plot."""
        # Find optimal parameters
        optimization_results = self.find_best_parameters(
            distribution, data, method, bounds
        )

        # Use dedicated plotting function
        return plot_parameter_optimization(
            data, optimization_results, distribution, **kwargs
        )

    def plot_distribution_comparison(
        self, data: np.ndarray, distributions: List[str] = None, **kwargs
    ):
        """Plot data histogram with multiple fitted distributions for comparison."""
        if distributions is None:
            distributions = list(self.distributions.keys())[
                :5
            ]  # Default to first 5 distributions

        # Fit parameters for each distribution
        fitted_params = {}
        for dist_name in distributions:
            try:
                if dist_name in self.distributions:
                    params = self.distributions[dist_name].fit(data)
                    fitted_params[dist_name] = params
            except:
                print(f"Could not fit {dist_name} distribution")

        # Use dedicated plotting function
        return plot_distribution_comparison(
            data, distributions, fitted_params, **kwargs
        )

    def plot_goodness_of_fit(
        self, data: np.ndarray, distribution: str, test: str = "ks", **kwargs
    ):
        """Plot goodness-of-fit test results with histogram, fitted PDF, and P-P plot."""
        # Perform goodness-of-fit test
        params = self.distributions[distribution].fit(data)

        if test.lower() == "ks":
            statistic, p_value = stats.kstest(data, distribution, args=params)
        elif test.lower() == "chi2":
            statistic, p_value = stats.chisquare(
                data, self.distributions[distribution].pdf(data, *params)
            )
        else:
            raise ValueError(f"Test '{test}' not supported")

        test_results = {
            "test": test,
            "parameters": params,
            "statistic": statistic,
            "p_value": p_value,
        }

        # Use dedicated plotting function
        return plot_goodness_of_fit(data, distribution, test_results, **kwargs)

    def plot_confidence_intervals(
        self,
        data: np.ndarray,
        distribution: str,
        confidence_level: float = 0.95,
        method: str = "bootstrap",
        n_bootstrap: int = 1000,
        **kwargs,
    ):
        """Plot confidence intervals for distribution parameters."""
        # Calculate confidence intervals
        import numpy as np
        from scipy import stats

        # Fit distribution
        params = self.distributions[distribution].fit(data)

        # Simple bootstrap for confidence intervals
        if method == "bootstrap":
            bootstrap_params = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                try:
                    bootstrap_params.append(
                        self.distributions[distribution].fit(sample)
                    )
                except:
                    continue

            bootstrap_params = np.array(bootstrap_params)

            # Calculate confidence intervals
            confidence_results = {
                "parameters": {},
                "confidence_level": confidence_level,
                "method": method,
                "sample_size": len(data),
            }

            for i in range(len(params)):
                param_name = f"param_{i}"
                lower = np.percentile(
                    bootstrap_params[:, i], (1 - confidence_level) / 2 * 100
                )
                upper = np.percentile(
                    bootstrap_params[:, i], (1 + confidence_level) / 2 * 100
                )

                confidence_results["parameters"][param_name] = {
                    "estimate": params[i],
                    "lower": lower,
                    "upper": upper,
                }
        else:
            raise ValueError(f"Method '{method}' not supported")

        # Use dedicated plotting function
        return plot_confidence_intervals(
            data, distribution, confidence_results, **kwargs
        )

    def plot_probability_plot(
        self,
        data: np.ndarray,
        distribution: str = "normal",
        plot_type: str = "qq",
        **kwargs,
    ):
        """Create probability plots (QQ or PP) for data against a theoretical distribution."""
        # Fit distribution
        params = self.distributions[distribution].fit(data)

        # Use dedicated plotting function
        return plot_probability_plot(
            data, distribution, params, plot_type=plot_type, **kwargs
        )

    def monte_carlo_simulation(
        self,
        distribution: str,
        params: tuple,
        n_simulations: int = 10000,
        statistic: str = "mean",
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Perform Monte Carlo simulation for a given statistic."""
        if distribution not in self.distributions:
            raise ValueError(f"Distribution '{distribution}' not supported")

        dist = self.distributions[distribution]

        # Generate multiple samples and calculate statistic for each
        statistics = []
        for _ in range(n_simulations):
            sample = dist.rvs(*params, size=100)  # Fixed sample size of 100

            if statistic == "mean":
                stat_value = np.mean(sample)
            elif statistic == "median":
                stat_value = np.median(sample)
            elif statistic == "std":
                stat_value = np.std(sample, ddof=1)
            elif statistic == "max":
                stat_value = np.max(sample)
            elif statistic == "min":
                stat_value = np.min(sample)
            elif statistic == "range":
                stat_value = np.max(sample) - np.min(sample)
            elif statistic == "skewness":
                stat_value = stats.skew(sample)
            elif statistic == "kurtosis":
                stat_value = stats.kurtosis(sample)
            else:
                raise ValueError(f"Statistic '{statistic}' not supported")

            statistics.append(stat_value)

        statistics = np.array(statistics)

        return {
            "statistic": statistic,
            "mean": np.mean(statistics),
            "std": np.std(statistics),
            "min": np.min(statistics),
            "max": np.max(statistics),
            "median": np.median(statistics),
            "q25": np.percentile(statistics, 25),
            "q75": np.percentile(statistics, 75),
            "distribution": statistics,
        }

    def bootstrap_analysis(
        self,
        data: np.ndarray,
        n_bootstrap: int = 1000,
        statistic: str = "mean",
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """Perform bootstrap analysis for confidence intervals."""
        n = len(data)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)

            if statistic == "mean":
                stat_value = np.mean(bootstrap_sample)
            elif statistic == "median":
                stat_value = np.median(bootstrap_sample)
            elif statistic == "std":
                stat_value = np.std(bootstrap_sample, ddof=1)
            elif statistic == "skewness":
                stat_value = stats.skew(bootstrap_sample)
            elif statistic == "kurtosis":
                stat_value = stats.kurtosis(bootstrap_sample)
            else:
                raise ValueError(f"Statistic '{statistic}' not supported")

            bootstrap_stats.append(stat_value)

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        return {
            "original_statistic": (
                getattr(np, statistic)(data)
                if statistic in ["mean", "median", "std"]
                else getattr(stats, statistic)(data)
            ),
            "bootstrap_mean": np.mean(bootstrap_stats),
            "bootstrap_std": np.std(bootstrap_stats),
            "confidence_interval": (ci_lower, ci_upper),
            "bias": np.mean(bootstrap_stats)
            - (
                getattr(np, statistic)(data)
                if statistic in ["mean", "median", "std"]
                else getattr(stats, statistic)(data)
            ),
        }

    def hypothesis_test(
        self, data: np.ndarray, test_type: str, **kwargs
    ) -> Dict[str, Union[float, str]]:
        """Expanded hypothesis testing capabilities."""
        results = {}

        if test_type == "normality":
            # Shapiro-Wilk test
            stat, p_val = stats.shapiro(data)
            results["shapiro_wilk"] = {"statistic": stat, "p_value": p_val}

            # Kolmogorov-Smirnov test for normality
            stat, p_val = stats.kstest(data, "norm", args=(np.mean(data), np.std(data)))
            results["ks_normality"] = {"statistic": stat, "p_value": p_val}

            # Anderson-Darling test
            try:
                stat, critical_vals, significance_levels = stats.anderson(
                    data, dist="norm"
                )
                results["anderson_darling"] = {
                    "statistic": stat,
                    "critical_values": critical_vals,
                    "significance_levels": significance_levels,
                }
            except:
                results["anderson_darling"] = {
                    "error": "Could not perform Anderson-Darling test"
                }

            # Jarque-Bera test
            stat, p_val = stats.jarque_bera(data)
            results["jarque_bera"] = {"statistic": stat, "p_value": p_val}

        elif test_type == "one_sample_t":
            mu = kwargs.get("mu", 0)
            # Get the alternative hypothesis type, default to "two-sided"
            alternative = kwargs.get("alternative", "two-sided")
            # Pass the 'alternative' argument to the t-test function
            stat, p_val = stats.ttest_1samp(data, mu, alternative=alternative)
            results["one_sample_t"] = {
                "statistic": stat,
                "p_value": p_val,
                "null_hypothesis": f"μ = {mu}",
                "alternative": alternative,
            }

        elif test_type == "two_sample_t":
            data2 = kwargs.get("data2", None)
            if data2 is None:
                raise ValueError("data2 required for two-sample t-test")

            equal_var = kwargs.get("equal_var", True)
            stat, p_val = stats.ttest_ind(data, data2, equal_var=equal_var)
            results["two_sample_t"] = {
                "statistic": stat,
                "p_value": p_val,
                "equal_variance_assumed": equal_var,
            }

        elif test_type == "paired_t":
            data2 = kwargs.get("data2", None)
            if data2 is None:
                raise ValueError("data2 required for paired t-test")

            stat, p_val = stats.ttest_rel(data, data2)
            results["paired_t"] = {"statistic": stat, "p_value": p_val}

        elif test_type == "chi_square_gof":
            observed = kwargs.get("observed", None)
            expected = kwargs.get("expected", None)
            if observed is not None and expected is not None:
                stat, p_val = stats.chisquare(observed, expected)
                results["chi_square_gof"] = {"statistic": stat, "p_value": p_val}
            else:
                # Perform goodness of fit test against uniform distribution
                hist, bin_edges = np.histogram(data, bins=kwargs.get("bins", 10))
                expected_freq = len(data) / len(hist)
                stat, p_val = stats.chisquare(hist, [expected_freq] * len(hist))
                results["chi_square_gof"] = {"statistic": stat, "p_value": p_val}

        elif test_type == "levene":
            # Test for equal variances
            data2 = kwargs.get("data2", None)
            if data2 is None:
                raise ValueError("data2 required for Levene test")

            stat, p_val = stats.levene(data, data2)
            results["levene"] = {"statistic": stat, "p_value": p_val}

        elif test_type == "mann_whitney":
            # Non-parametric test for comparing two independent samples
            data2 = kwargs.get("data2", None)
            if data2 is None:
                raise ValueError("data2 required for Mann-Whitney test")

            stat, p_val = stats.mannwhitneyu(data, data2)
            results["mann_whitney"] = {"statistic": stat, "p_value": p_val}

        elif test_type == "wilcoxon":
            # Non-parametric test for comparing paired samples
            data2 = kwargs.get("data2", None)
            if data2 is not None:
                stat, p_val = stats.wilcoxon(data, data2)
            else:
                # One-sample Wilcoxon test
                stat, p_val = stats.wilcoxon(data)
            results["wilcoxon"] = {"statistic": stat, "p_value": p_val}

        elif test_type == "kruskal_wallis":
            # Non-parametric test for comparing multiple groups
            groups = kwargs.get("groups", [])
            if len(groups) < 2:
                raise ValueError("At least 2 groups required for Kruskal-Wallis test")

            stat, p_val = stats.kruskal(*groups)
            results["kruskal_wallis"] = {"statistic": stat, "p_value": p_val}

        elif test_type == "runs":
            # Test for randomness
            # Convert to binary based on median
            median_val = np.median(data)
            binary_data = (data > median_val).astype(int)

            # Count runs
            runs = 1
            for i in range(1, len(binary_data)):
                if binary_data[i] != binary_data[i - 1]:
                    runs += 1

            n1 = np.sum(binary_data)
            n2 = len(binary_data) - n1

            if n1 > 0 and n2 > 0:
                expected_runs = 2 * n1 * n2 / (n1 + n2) + 1
                var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (
                    (n1 + n2) ** 2 * (n1 + n2 - 1)
                )

                if var_runs > 0:
                    z_stat = (runs - expected_runs) / np.sqrt(var_runs)
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                    results["runs"] = {
                        "observed_runs": runs,
                        "expected_runs": expected_runs,
                        "z_statistic": z_stat,
                        "p_value": p_val,
                    }

        return results

    def confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95,
        parameter: str = "mean",
        method: str = "parametric",
    ) -> Dict[str, Union[float, Tuple[float, float]]]:
        """Enhanced confidence interval calculation with multiple methods."""
        alpha = 1 - confidence
        n = len(data)

        results = {
            "confidence_level": confidence,
            "method": method,
            "parameter": parameter,
        }

        if method == "parametric":
            if parameter == "mean":
                mean = np.mean(data)
                std_err = stats.sem(data)
                t_val = stats.t.ppf(1 - alpha / 2, n - 1)
                margin_error = t_val * std_err
                ci = (mean - margin_error, mean + margin_error)

                results.update(
                    {
                        "point_estimate": mean,
                        "standard_error": std_err,
                        "margin_of_error": margin_error,
                        "confidence_interval": ci,
                    }
                )

            elif parameter == "std":
                std = np.std(data, ddof=1)
                chi2_lower = stats.chi2.ppf(alpha / 2, n - 1)
                chi2_upper = stats.chi2.ppf(1 - alpha / 2, n - 1)
                ci = (
                    np.sqrt((n - 1) * std**2 / chi2_upper),
                    np.sqrt((n - 1) * std**2 / chi2_lower),
                )

                results.update({"point_estimate": std, "confidence_interval": ci})

            elif parameter == "variance":
                var = np.var(data, ddof=1)
                chi2_lower = stats.chi2.ppf(alpha / 2, n - 1)
                chi2_upper = stats.chi2.ppf(1 - alpha / 2, n - 1)
                ci = ((n - 1) * var / chi2_upper, (n - 1) * var / chi2_lower)

                results.update({"point_estimate": var, "confidence_interval": ci})

            elif parameter == "proportion":
                # Assuming data is binary (0s and 1s)
                if not np.all(np.isin(data, [0, 1])):
                    raise ValueError(
                        "Data must be binary (0s and 1s) for proportion CI"
                    )

                p_hat = np.mean(data)
                std_err = np.sqrt(p_hat * (1 - p_hat) / n)
                z_val = stats.norm.ppf(1 - alpha / 2)
                margin_error = z_val * std_err
                ci = (max(0, p_hat - margin_error), min(1, p_hat + margin_error))

                results.update(
                    {
                        "point_estimate": p_hat,
                        "standard_error": std_err,
                        "margin_of_error": margin_error,
                        "confidence_interval": ci,
                    }
                )

        elif method == "bootstrap":
            bootstrap_result = self.bootstrap_analysis(
                data, n_bootstrap=1000, statistic=parameter, confidence_level=confidence
            )
            results.update(
                {
                    "point_estimate": bootstrap_result["original_statistic"],
                    "confidence_interval": bootstrap_result["confidence_interval"],
                    "bootstrap_mean": bootstrap_result["bootstrap_mean"],
                    "bootstrap_std": bootstrap_result["bootstrap_std"],
                    "bias": bootstrap_result["bias"],
                }
            )

        return results

    def power_analysis(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = None,
        n: int = None,
        test_type: str = "one_sample_t",
    ) -> Dict[str, float]:
        """
        Perform power analysis for different types of tests.
        Can calculate required sample size (if power is provided) or
        can calculate power (if n is provided).
        """
        if power is None and n is None:
            raise ValueError("Either 'power' or 'n' must be provided.")
        if power is not None and n is not None:
            raise ValueError("Provide either 'power' or 'n', not both.")

        from scipy.stats import t
        from scipy.optimize import minimize_scalar

        results = {
            "test_type": test_type,
            "effect_size": effect_size,
            "alpha": alpha,
        }

        if test_type in ["one_sample_t", "paired_t"]:

            def power_function(sample_size):
                if sample_size < 2:
                    return 0.0
                df = sample_size - 1
                t_critical = t.ppf(1 - alpha / 2, df)
                ncp = effect_size * np.sqrt(sample_size)
                return 1 - (t.cdf(t_critical, df, ncp) - t.cdf(-t_critical, df, ncp))

            if n:  # Calculate power for a given n
                calculated_power = power_function(n)
                results.update({"sample_size": n, "power": calculated_power})
            else:  # Calculate n for a given power

                def objective(sample_size):
                    return (power_function(sample_size) - power) ** 2

                res = minimize_scalar(objective, bounds=(2, 10000), method="bounded")
                required_n = int(np.ceil(res.x))
                actual_power = power_function(required_n)
                results.update(
                    {
                        "desired_power": power,
                        "required_sample_size": required_n,
                        "actual_power": actual_power,
                    }
                )

        elif test_type == "two_sample_t":

            def power_function(n_per_group):
                if n_per_group < 2:
                    return 0.0
                df = 2 * n_per_group - 2
                t_critical = t.ppf(1 - alpha / 2, df)
                ncp = effect_size * np.sqrt(n_per_group / 2)
                return 1 - (t.cdf(t_critical, df, ncp) - t.cdf(-t_critical, df, ncp))

            if n:  # Calculate power for total sample size n
                n_per_group = n / 2
                calculated_power = power_function(n_per_group)
                results.update({"total_sample_size": n, "power": calculated_power})
            else:  # Calculate n for a given power

                def objective(n_per_group):
                    return (power_function(n_per_group) - power) ** 2

                res = minimize_scalar(objective, bounds=(2, 10000), method="bounded")
                required_n_per_group = int(np.ceil(res.x))
                actual_power = power_function(required_n_per_group)
                results.update(
                    {
                        "desired_power": power,
                        "required_sample_size_per_group": required_n_per_group,
                        "total_sample_size": 2 * required_n_per_group,
                        "actual_power": actual_power,
                    }
                )

        else:
            raise ValueError(f"Power analysis for {test_type} not implemented")

        return results

    def outlier_detection(
        self, data: np.ndarray, method: str = "iqr", threshold: float = 1.5
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Detect outliers using various methods."""

        results = {"method": method}

        if method == "iqr":
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]

            results.update(
                {
                    "outliers": outliers,
                    "outlier_indices": outlier_indices,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "num_outliers": len(outliers),
                    "outlier_percentage": len(outliers) / len(data) * 100,
                }
            )

        elif method == "z_score":
            mean = np.mean(data)
            std = np.std(data)
            z_scores = np.abs((data - mean) / std)

            outliers = data[z_scores > threshold]
            outlier_indices = np.where(z_scores > threshold)[0]

            results.update(
                {
                    "outliers": outliers,
                    "outlier_indices": outlier_indices,
                    "z_scores": z_scores,
                    "threshold": threshold,
                    "num_outliers": len(outliers),
                    "outlier_percentage": len(outliers) / len(data) * 100,
                }
            )

        elif method == "modified_z_score":
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad

            outliers = data[np.abs(modified_z_scores) > threshold]
            outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]

            results.update(
                {
                    "outliers": outliers,
                    "outlier_indices": outlier_indices,
                    "modified_z_scores": modified_z_scores,
                    "threshold": threshold,
                    "num_outliers": len(outliers),
                    "outlier_percentage": len(outliers) / len(data) * 100,
                }
            )

        elif method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest

                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(data.reshape(-1, 1))

                outliers = data[outlier_labels == -1]
                outlier_indices = np.where(outlier_labels == -1)[0]

                results.update(
                    {
                        "outliers": outliers,
                        "outlier_indices": outlier_indices,
                        "outlier_labels": outlier_labels,
                        "num_outliers": len(outliers),
                        "outlier_percentage": len(outliers) / len(data) * 100,
                    }
                )

            except ImportError:
                results["error"] = "sklearn not available for isolation forest method"

        return results

    def distribution_transformation(
        self, data: np.ndarray, transformation: str = "log"
    ) -> Dict[str, Union[np.ndarray, str]]:
        """Apply transformations to make data more normal."""

        results = {"transformation": transformation}

        if transformation == "log":
            if np.any(data <= 0):
                # Add small constant to handle zeros/negatives
                transformed_data = np.log(data - np.min(data) + 1e-10)
                results["note"] = "Added constant to handle non-positive values"
            else:
                transformed_data = np.log(data)

        elif transformation == "sqrt":
            if np.any(data < 0):
                transformed_data = np.sqrt(data - np.min(data))
                results["note"] = "Shifted data to handle negative values"
            else:
                transformed_data = np.sqrt(data)

        elif transformation == "reciprocal":
            if np.any(data == 0):
                transformed_data = 1 / (data + 1e-10)
                results["note"] = "Added small constant to handle zeros"
            else:
                transformed_data = 1 / data

        elif transformation == "square":
            transformed_data = data**2

        elif transformation == "box_cox":
            from scipy.stats import boxcox

            if np.any(data <= 0):
                shifted_data = data - np.min(data) + 1
                transformed_data, lambda_param = boxcox(shifted_data)
                results["note"] = "Data was shifted to be positive"
            else:
                transformed_data, lambda_param = boxcox(data)
            results["lambda"] = lambda_param

        elif transformation == "yeo_johnson":
            from scipy.stats import yeojohnson

            transformed_data, lambda_param = yeojohnson(data)
            results["lambda"] = lambda_param

        elif transformation == "standardize":
            transformed_data = (data - np.mean(data)) / np.std(data)

        elif transformation == "normalize":
            transformed_data = (data - np.min(data)) / (np.max(data) - np.min(data))

        else:
            raise ValueError(f"Transformation '{transformation}' not supported")

        # Calculate normality before and after transformation
        _, p_before = stats.shapiro(data)
        _, p_after = stats.shapiro(transformed_data)

        results.update(
            {
                "original_data": data,
                "transformed_data": transformed_data,
                "normality_p_before": p_before,
                "normality_p_after": p_after,
                "improvement": p_after > p_before,
            }
        )

        return results

    def generate_report(
        self, data: np.ndarray, distributions_to_test: List[str] = None
    ) -> str:
        """Generate a comprehensive analysis report."""

        if distributions_to_test is None:
            distributions_to_test = [
                "normal",
                "lognormal",
                "exponential",
                "gamma",
                "weibull",
            ]

        report = "=" * 60 + "\n"
        report += "COMPREHENSIVE DISTRIBUTION ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"

        # Basic statistics
        report += "1. DESCRIPTIVE STATISTICS\n"
        report += "-" * 30 + "\n"
        stats_dict = self.calculate_statistics(data)
        for key, value in stats_dict.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"

        # Distribution comparison
        report += "\n2. DISTRIBUTION FITTING RESULTS\n"
        report += "-" * 30 + "\n"
        comparison_df = self.compare_distributions(data, distributions_to_test)

        if not comparison_df.empty:
            report += "Best fitting distributions (ranked by AIC):\n"
            for idx, row in comparison_df.head(3).iterrows():
                report += (
                    f"{idx+1}. {row['distribution'].title()}: AIC={row['aic']:.2f}, "
                )
                report += f"BIC={row['bic']:.2f}, KS p-value={row['ks_p_value']:.4f}\n"

        # Normality tests
        report += "\n3. NORMALITY TESTS\n"
        report += "-" * 30 + "\n"
        normality_results = self.hypothesis_test(data, "normality")

        for test_name, result in normality_results.items():
            if "p_value" in result:
                report += f"{test_name.replace('_', ' ').title()}: "
                report += f"p-value = {result['p_value']:.4f} "
                report += f"({'Normal' if result['p_value'] > 0.05 else 'Not Normal'} at α=0.05)\n"

        # Outlier detection
        report += "\n4. OUTLIER ANALYSIS\n"
        report += "-" * 30 + "\n"
        outlier_results = self.outlier_detection(data, method="iqr")
        report += f"IQR Method: {outlier_results['num_outliers']} outliers detected "
        report += f"({outlier_results['outlier_percentage']:.1f}% of data)\n"

        outlier_results_z = self.outlier_detection(data, method="z_score", threshold=3)
        report += (
            f"Z-Score Method: {outlier_results_z['num_outliers']} outliers detected "
        )
        report += f"({outlier_results_z['outlier_percentage']:.1f}% of data)\n"

        # Recommendations
        report += "\n5. RECOMMENDATIONS\n"
        report += "-" * 30 + "\n"

        best_dist = (
            comparison_df.iloc[0]["distribution"]
            if not comparison_df.empty
            else "unknown"
        )
        report += f"• Best fitting distribution: {best_dist.title()}\n"

        if normality_results["shapiro_wilk"]["p_value"] < 0.05:
            report += "• Data is not normally distributed - consider transformations\n"

            # Test transformations
            log_result = self.distribution_transformation(data, "log")
            if log_result["improvement"]:
                report += "• Log transformation may improve normality\n"
        else:
            report += "• Data appears to be normally distributed\n"

        if outlier_results["num_outliers"] > 0:
            report += f"• {outlier_results['num_outliers']} potential outliers detected - investigate further\n"

        # Confidence intervals
        report += "\n6. CONFIDENCE INTERVALS (95%)\n"
        report += "-" * 30 + "\n"
        ci_mean = self.confidence_interval(data, 0.95, "mean")
        ci_std = self.confidence_interval(data, 0.95, "std")

        report += f"Mean: {ci_mean['point_estimate']:.4f} "
        report += f"[{ci_mean['confidence_interval'][0]:.4f}, {ci_mean['confidence_interval'][1]:.4f}]\n"
        report += f"Standard Deviation: {ci_std['point_estimate']:.4f} "
        report += f"[{ci_std['confidence_interval'][0]:.4f}, {ci_std['confidence_interval'][1]:.4f}]\n"

        report += "\n" + "=" * 60 + "\n"
        report += "Report generated using DistributionAnalyzer\n"
        report += "=" * 60 + "\n"

        return report

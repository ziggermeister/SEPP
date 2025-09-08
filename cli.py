import argparse

from ingest.features import annualized_mu_sigma, dividend_yield_from_prices
from ingest.multiprovider_prices import get_prices_multi
from portfolio.schema import load_portfolio
from portfolio.visualize import plot_portfolio_prices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--portfolio_json",
        default=None,
        help='{"name":"p","weights":{"VTI":0.6,"SGOV":0.4}}',
    )
    args = ap.parse_args()

    df = get_prices_multi(args.symbols, args.start, args.end, "1d")
    print("Fetched columns:", df.columns[:6], " ...")
    mu, sig, rho = annualized_mu_sigma(df)
    _ = dividend_yield_from_prices(df)
    print("Estimated mu:", mu.round(4))
    print("Estimated sig:", sig.round(4))
    print("Estimated rho[0,:5]:", rho[0, :5].round(3))

    if args.portfolio_json:
        p = load_portfolio(args.portfolio_json)
        plot_portfolio_prices(df, p.weights, title=f"Portfolio: {p.name}")


if __name__ == "__main__":
    main()

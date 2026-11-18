"""Step 0: split a domain into subdomain, name and suffix with tldextract."""

# pandas: the library for tables (DataFrames). Our dataset is a table.
import pandas as pd

# tldextract: the library that knows which endings are real suffixes
# (e.g. "com", "co.uk", "org.lb"), so it can split domains correctly.
import tldextract

# Endings used by some DGA families that are NOT in the official suffix list
# (they belong to alternative DNS systems such as Namecoin, EmerDNS, OpenNIC),
# so tldextract does not know them. Without them, tldextract returns no suffix
# and puts the ending in the name (e.g. 'edwiudli.bazar' -> name 'bazar').
# Found by listing the domains with an empty suffix (see notes/findings.md):
#   .bazar, .bazaar -> bazarbackdoor
#   .bit            -> necurs
#   .load           -> gozi
#   .oss, .geek, .web -> other_dga
EXTRA_SUFFIXES = ["bazar", "bazaar", "bit", "load", "oss", "geek", "web"]

# Create one "splitter" object that we reuse for every domain.
# suffix_list_urls=() : use the suffix list bundled with tldextract 5.4.0
#                       (no internet download), so everyone gets the same split.
# extra_suffixes      : add the DGA endings above to that list.
_extract = tldextract.TLDExtract(
    suffix_list_urls=(),
    extra_suffixes=EXTRA_SUFFIXES,
    include_psl_private_domains=True,   # treat ddns.net, duckdns.org, ... as suffixes
)


def split_domain(domain: str) -> tuple[str, str, str]:
    """Split ONE domain into (subdomain, name, suffix).

    Example: 'mail.bbc.co.uk' -> ('mail', 'bbc', 'co.uk')
    """
    # tldextract does the split.
    parts = _extract(domain)
    sub, name, suffix = parts.subdomain, parts.domain, parts.suffix

    # Special case (8 legit domains in train): the whole domain is itself an
    # official suffix (e.g. 'gov.bw', 'ac.pk'), so tldextract leaves the name
    # empty. We use the first label of the suffix as the name, so the name
    # features are not computed on an empty string.
    #   'gov.bw' -> ('', 'gov', 'bw')
    # TODO: confirm with the team (alternative: drop these rows).
    if name == "":
        first, _, rest = suffix.partition(".")
        name, suffix = first, rest

    return sub, name, suffix


def add_split_columns(df: pd.DataFrame, col: str = "domain") -> pd.DataFrame:
    """Split EVERY domain in a table and add four new columns.

    df  : the table of domains (e.g. our 2M-row dataset)
    col : the name of the column that holds the domains (default: "domain")
    """
    # .map(split_domain) runs split_domain on every row of the column.
    # Each cell becomes a tuple, e.g. ('', 'google', 'com').
    parts = df[col].map(split_domain)

    # Work on a copy so the original table passed in is not changed.
    out = df.copy()

    # .str[0] takes the 1st item of each tuple, .str[1] the 2nd, .str[2] the 3rd.
    out["subdomain"] = parts.str[0]   # e.g. 'mail', or '' if there is none
    out["name"] = parts.str[1]        # e.g. 'google'
    out["suffix"] = parts.str[2]      # e.g. 'com', 'co.uk' -> used for TLD features

    # 'text' = everything before the suffix, without dots. The character features
    # (n-grams, Markov, vowels) are computed on this, so a random part sitting in
    # the subdomain is not missed:
    #   'mail.google.com'        -> 'mailgoogle'
    #   'cf56c91076.session.oss' -> 'cf56c91076session'
    out["text"] = (out["subdomain"] + out["name"]).str.replace(".", "", regex=False)

    return out
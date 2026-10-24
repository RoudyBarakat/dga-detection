# Findings

Things we discovered about the data, for the report and presentation.

- 2026-10-24 (Abdo): tldextract does not know endings from alternative DNS systems
  used by some DGA families: .bazar/.bazaar (bazarbackdoor), .bit (necurs),
  .load (gozi), .oss/.geek/.web (other_dga). 23,743 train domains, all DGA, got an
  empty suffix and the ending as their name (e.g. edwiudli.bazar -> name "bazar").
  Fixed by passing these endings to tldextract (extra_suffixes, see
  src/features/split.py). 8 legit domains are themselves official suffixes
  (gov.bw, ac.pk...) and got an empty name; we use the first label as the name
  (to confirm). These TLDs appear only in DGA domains, so the TLD alone
  identifies these families: one reason for the with/without-TLD ablation.

- 2026-10-24 (Abdo): many DGA domains live under free dynamic-DNS services
  (ddns.net, duckdns.org, endofinternet.net...), so tldextract treated the service
  as the name and the random part as a subdomain. Fixed with tldextract's private
  suffix list (include_psl_private_domains=True). Services not in that list
  (co.cc, cz.cc, dynserv.com...) are covered by the 'text' column (subdomain + name,
  no dots), on which all character features are computed.

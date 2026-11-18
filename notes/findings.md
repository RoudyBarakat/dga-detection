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

- 2026-10-25 (Abdo): with ~800k legit training names, almost every 2-gram and 3-gram
  appears at least once, so "share of n-grams never seen in legit names" was ~0 for
  everyone (useless). Replaced it with "share of rare n-grams" (less than 1 in a
  million). For 3-grams this works (DGA 19.6% vs legit 1.6%); for 2-grams it is still
  always 0 (only ~1,400 possible 2-grams, all common), so ng2_rare is dropped.
  N-gram features are the strongest single features so far (ng3_mean AUC ~0.92).

- 2026-11-01 (Abdo): switched the out-of-fold n-gram and Markov features from 5 to
  10 folds (training rows now scored with 90% of the counts instead of 80%). The
  train/val gap in the leakage check shrank (DGA ng3_rare: 0.196/0.183 -> 0.178/0.183).

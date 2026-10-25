\# Findings



Things we discovered about the data, for the report and presentation.



\- 2026-10-24 (Abdo): tldextract does not know endings from alternative DNS systems

&#x20; used by some DGA families: .bazar/.bazaar (bazarbackdoor), .bit (necurs),

&#x20; .load (gozi), .oss/.geek/.web (other\_dga). 23,743 train domains, all DGA, got an

&#x20; empty suffix and the ending as their name (e.g. edwiudli.bazar -> name "bazar").

&#x20; Fixed by passing these endings to tldextract (extra\_suffixes, see

&#x20; src/features/split.py). 8 legit domains are themselves official suffixes

&#x20; (gov.bw, ac.pk...) and got an empty name; we use the first label as the name

&#x20; (to confirm). These TLDs appear only in DGA domains, so the TLD alone

&#x20; identifies these families: one reason for the with/without-TLD ablation.



\- 2026-10-24 (Abdo): many DGA domains live under free dynamic-DNS services

&#x20; (ddns.net, duckdns.org, endofinternet.net...), so tldextract treated the service

&#x20; as the name and the random part as a subdomain. Fixed with tldextract's private

&#x20; suffix list (include\_psl\_private\_domains=True). Services not in that list

&#x20; (co.cc, cz.cc, dynserv.com...) are covered by the 'text' column (subdomain + name,

&#x20; no dots), on which all character features are computed.



\- 2026-10-25 (Abdo): with \~800k legit training names, almost every 2-gram and 3-gram

&#x20; appears at least once, so "share of n-grams never seen in legit names" was \~0 for

&#x20; everyone (useless). Replaced it with "share of rare n-grams" (less than 1 in a

&#x20; million). For 3-grams this works (DGA 19.6% vs legit 1.6%); for 2-grams it is still

&#x20; always 0 (only \~1,400 possible 2-grams, all common), so ng2\_rare is dropped.

&#x20; N-gram features are the strongest single features so far (ng3\_mean AUC \~0.92).


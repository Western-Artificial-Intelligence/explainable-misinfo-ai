\# Data Inconsistencies/Problems \& Heads Up for Dev 2



Just some stuff to watch out for when you're cleaning the data.



\## LIAR



The labels are 6-class (true, mostly-true, half-true, barely-true, false, pants-fire) and you'll need to squash them down to 3. Probably makes sense to group:

\- 0 = true + mostly-true (the good ones)

\- 1 = half-true + barely-true (the middle ground)

\- 2 = false + pants-fire (the bad ones)



The statements themselves are pretty short and clean which is good. But some have really long URLs in the context field—you might want to just replace those with `\[URL]` or drop them, depending on what you're doing.



The count columns (barely\_true\_counts, false\_counts, etc.) are stored as strings, not numbers, so parsing them is kinda i don't know difficult.





\## CoAID



This one's a bit messier because it's organized as 4 snapshots (May, July, Sept, Nov 2020) with separate files for claims, news, tweets, tweet replies. Each snapshot has slightly different files.



The labels are kinda implicit in the filenames (ClaimFakeCOVID-19.csv means "fake", NewsRealCOVID-19.csv means "real"). The CSV columns might not always have an explicit label field, so you'll need to figure out the label from the filename.



Tweet text has all the usual social media stuff—hashtags, mentions, emojis, shortened URLs. There's also some non-ASCII and multilingual stuff



The engagement metrics (retweets, favorites, replies) are just snapshots from those dates, not cumulative. They just show how popular something was at that moment.



\## FakeHealth



Heads up: \*\*there are no labels in the extracted data\*\*.

The reviews and engagements folders were empty when we cloned the repo. So you've got 2,237 health articles but nothing saying whether they're true or false. You'll need to either:

\- Re-clone the full repo and see if those folders have data

\- Manually label some for validation



The articles are long (average ~4k chars, some over 25k), so if you're feeding them into a model, you might want to chunk or truncate them.



Content types are split unevenly: HealthStory (1,638) vs. HealthRelease (599).



\## General Stuff



When you're mapping everything to 3-class, keep the original labels in your output CSV too. Makes it way easier to debug later.



Text is generally pretty clean across all three, minimal HTML artifacts. Watch for special characters and emoji in CoAID tweets.


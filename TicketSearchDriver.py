import numpy as np
import pandas as pd
import timeit


from myTicketMethod_KMeans import SearchTickets

df = pd.read_csv('TicketsData.csv')

start_time = timeit.default_timer()

searchTickets = SearchTickets(k_tickets = 30, scale = 10)

elapsed = timeit.default_timer() - start_time
print('Init Time:')
print(elapsed)






start_time = timeit.default_timer()

searchTickets.fit(df)
searchTickets.recluster()

elapsed = timeit.default_timer() - start_time
print('Fit Time:')
print(elapsed)



search = 'Show me software update issues with nirc2.'
#search = 'I want to see tickets with the insturment osiris and its ao problems.'
#search = 'im not sure what is about to happen.'
#search = 'azimuth pointing ca arcsec'
#search = 'mosfire'
search = 'show me weather loss'
start_time = timeit.default_timer()

result = searchTickets.search(search)
elapsed = timeit.default_timer() - start_time
print('Search Time:')
print(elapsed)

print('Search Result:')
print(result)

print(len(searchTickets.groups), len(df))
df = searchTickets.df
df['Groups'] = searchTickets.groups
df.to_csv(r'TicketsData_grouped.csv', header=True)

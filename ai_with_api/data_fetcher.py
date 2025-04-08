import requests
import pandas as pd
import time

API_KEY = '331311'
BASE_URL = 'https://www.thesportsdb.com/api/v1/json/331311'
LEAGUE_ID = '4335'  # English Premier League ID

league_name = 'LaLiga'  # Replace with the appropriate league name variable
start_year = 2000
end_year = 2024

# League IDs
# English Premier League: 4328
# Spanish La Liga: 4335
# German Bundesliga: 4331
# Italian Serie A: 4332
# French Ligue 1: 4334
# Portuguese Primeira Liga: 4344
# Dutch Eredivisie: 4337
# Scottish Premiership: 4330
# English Championship: 4329
# Major League Soccer (MLS): 4346
# Brazilian Série A: 4351
# Argentine Primera División: 4406
# Mexican Liga MX: 4350
# Japanese J1 League: 4410
# Turkish Süper Lig: 4339
# Polish Ekstraklasa: 4422


SEASON = '2024-2025'
SEASON1 = '2023-2024'
SEASON2 = '2022-2023'
SEASON3 = '2021-2022'
SEASON4 = '2020-2021'
SEASON5 = '2019-2020'
SEASON6 = '2018-2019'
SEASON7 = '2017-2018'
SEASON8 = '2016-2017'
SEASON9 = '2015-2016'
SEASON10 = '2014-2015'
SEASON11 = '2013-2014'
SEASON12 = '2012-2013'
SEASON13 = '2011-2012'
SEASON14 = '2010-2011'
SEASON15 = '2009-2010'
SEASON16 = '2008-2009'
SEASON17 = '2007-2008'
SEASON18 = '2006-2007'
SEASON19 = '2005-2006'
SEASON20 = '2004-2005'
SEASON21 = '2003-2004'
SEASON22 = '2002-2003'
SEASON23 = '2001-2002'
SEASON24 = '2000-2001'

url = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON}'
url_1 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON1}'
url_2 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON2}'
url_3 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON3}'
url_4 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON4}'
url_5 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON5}'
url_6 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON6}'
url_7 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON7}'
url_8 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON8}'
url_9 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON9}'
url_10 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON10}'
url_11 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON11}'
url_12 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON12}'
url_13 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON13}'
url_14 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON14}'
url_15 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON15}'
url_16 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON16}'
url_17 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON17}'
url_18 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON18}'
url_19 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON19}'
url_20 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON20}'
url_21 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON21}'
url_22 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON22}'
url_23 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON23}'
url_24 = f'{BASE_URL}/eventsseason.php?id={LEAGUE_ID}&s={SEASON24}'

all_matches = []

seasons_and_urls = [
    (url, SEASON),
    (url_1, SEASON1),
    (url_2, SEASON2),
    (url_3, SEASON3),
    (url_4, SEASON4),
    (url_5, SEASON5),
    (url_6, SEASON6),
    (url_7, SEASON7),
    (url_8, SEASON8),
    (url_9, SEASON9),
    (url_10, SEASON10),
    (url_11, SEASON11),
    (url_12, SEASON12),
    (url_13, SEASON13),
    (url_14, SEASON14),
    (url_15, SEASON15),
    (url_16, SEASON16),
    (url_17, SEASON17),
    (url_18, SEASON18),
    (url_19, SEASON19),
    (url_20, SEASON20),
    (url_21, SEASON21),
    (url_22, SEASON22),
    (url_23, SEASON23),
    (url_24, SEASON24),
]

for url, season in seasons_and_urls:
    print(f"Fetching data for season {season}...")
    response = requests.get(url)

    if response.status_code == 200:
        events = response.json().get('events', [])
        if events:
            df = pd.DataFrame(events)
            columns = ['idEvent', 'strEvent', 'dateEvent', 'strTime',
                       'strHomeTeam', 'strAwayTeam', 'intHomeScore', 'intAwayScore']
            df = df[columns]
            df['season'] = season
            all_matches.append(df)
            print(f'Fetched {len(df)} matches for season {season}')
        else:
            print(f'No data found for season {season}')
    else:
        print(f'Error fetching data for season {season}: {response.status_code}')

    time.sleep(1)  # Respect API rate limits

if all_matches:
    output_filename = f'{league_name}_matches_{start_year}-{end_year}.csv'
    final_df = pd.concat(all_matches, ignore_index=True)
    final_df.to_csv(output_filename, index=False)
    print(f'Saved {len(final_df)} matches to {output_filename}')
else:
    print('No data was fetched')

```python
#import libararies
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import urllib.parse
from urllib.parse import urljoin
import re
```

# Save the base links of TripAdvisor and create lists to store links later


```python
urlmain = "https://www.tripadvisor.com/Hotels-g191-United_States-Hotels.html" # main page of U.S hotels 
base_url = "https://www.tripadvisor.com" # base link of the TripAdvisor website

list_local = [] # store url for the first pages of top 100 U.S cities of hotel services
full_list_local = [] # store url for first 5 pages of top 100 cities # 500 pages in the total
list_hotel = [] # store url for hotels in these 500 pages

```

# Retrieve the hotel links

## Step 1: Scrape 100 location links

* I will scrape the links of 100 locations (5 pages, 20 locations/page). <br> 
* Due to the different structure of the first page, I have to write a separate code for it. <br> 
* For other pages, I will use the for loop to retrieve the links. <br> 
* The 100 location links will be stored in the list_local. 


```python
# retrieve the links in the first page and store them in list_local
html = requests.get(urlmain)
soup = BeautifulSoup(html.content, 'lxml')
for item in soup.find_all("a", attrs={"class":"linkText"}):
    full_loc_url = []
    loc_url = item.get("href") # get all the href containing the links for the location
    full_loc_url = urljoin(base_url,loc_url) # join the href links with the base link of TripAdvisor
    list_local.append(full_loc_url) # store the links 

```


```python
# retrieve the links in the next 4 pages
urlhead = "https://www.tripadvisor.com/Hotels-g191-oa" # separate the links into two parts
urltail = "-United_States-Hotels.html#LEAF_GEO_LIST"   # the number between two urls will determine the page
for i in range (20,100,20): # i receives the values: 20,40,60,80,... corresponding to pages: 2,3,4,5,...
    url = urlhead + str(i) + urltail # the url of the page I try to scrape
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'lxml')
    for item in soup.find_all("a", attrs={"class":"city"}):
        full_loc_url = []
        loc_url = item.get("href")
        full_loc_url = urljoin(base_url,loc_url)
        list_local.append(full_loc_url) # store the links in the list_local

```


```python
len(list_local) # check if I get all the 100 links
#list_local[:10] 
```




    100



## Step 2: Retrieve the links for the first 5 pages of each location

* I got the 100 location links but the links represent their first pages only.
* For each location, I want to hotel links in the first 5 pages. There will be 500 pages in the total.
* The code below will retrieve 500 pages and store them in full_list_local.


```python
split_char = "-" # split the links at the "-" character
                # the variable defines the page number will be at the 2nd "-" position

for url in list_local:
    temp = url.split(split_char)
    for i in range (0,150,30):
        # join the link again with the variable defining the page number in the middle
        page_url = split_char.join(temp[:2])+ "-oa" + str(i) + "-" + split_char.join(temp[2:]) 
        full_list_local.append(page_url)
    


   

```


```python
len(full_list_local) # check if I get all 500 pages
#full_list_local[:10]
```




    500



## Step 3: Scrape the hotel links

* I will write a code to scrape all the hotel links from 500 pages (~ 0 hotels per page).
* Theoretically, I would have around 15,000 hotels but actually there are many locations with under 100 hotels.
* The real number will be smaller than 15,000. I expect to have 10,000-13,000 hotels for my dataset.


```python
for url in full_list_local:
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'lxml')
    div = soup.find_all("div", attrs={"class":"listing_title"}) # find all the div(s) that contain the href for hotels
    
    for item in div:
        for i in item.find_all("a"):
            hotel_url = []
            href_url = i.get("href") # get the href(s)
            hotel_url = urljoin(base_url,href_url) # join href(s) with the base link
            list_hotel.append(hotel_url) # save the hotel links
```


```python
hotel_html = pd.Series(list_hotel).drop_duplicates().tolist() # drop duplicates and save them in hotel_html
```


```python
len(hotel_html) # check how many hotels I will have for my dataset
```




    12313



# Scrape the information from hotel pages

There are 9 attributes I want to scrape in each hotel page:
* The hotel name.
* The location/city of the hotel.
* Cost to stay at the hotel per night (excluding taxes and other fees).
* TripAdvisor Score for the hotel.
* Ratings for the hotel (excellent, good, bad,...).
* Walk grade (0-100): shows how convenient travelers feel when moving to places near the hotel.
* Number of nearby restaurants.
* Number of nearby attractions.
* State code of the hotel.





```python
# create 9 empty lists to store the data
# Some values are missing so I use the try-except here. Missing values are recorded as NAs.
name = []
location = []
price = []
score = []
walk = []
restaurant = []
attraction = []
label = []
state = []

for url in hotel_html:
    link = requests.get(url)
    soup = BeautifulSoup(link.content, 'lxml')
    
    #scrape the name
    try:
        nm = soup.find("h1", attrs={"class":"hotels-hotel-review-atf-info-parts-Heading__heading--2ZOcD"})
        name.append(nm.text)
    except:
        name.append("NA")
    
    #scrape the location
    try:
        loc = soup.find("a", attrs={"data-tracking-label":"tourism"})
        location.append(loc.text)
    except:
        location.append("NA")
    
    #scrape the price
    try:
        pc = soup.find("div", attrs={"data-sizegroup":"hr_chevron_prices"})
        price.append(pc.text)
    except:
        try:
            pc = soup.find("div", attrs={"class":"hotels-hotel-offers-DominantOffer__price--D-ycN"})
            price.append(pc.text)
        except:
            price.append("NA")
    
    #scrape the score
    try:
        scr = soup.find("span", attrs={"class":"hotels-hotel-review-about-with-photos-Reviews__overallRating--vElGA"})
        score.append(scr.text)
    except:
        score.append("NA")
    
    #scrape the ratings
    try:
        lbl = soup.find("div", attrs={"class":"hotels-hotel-review-about-with-photos-Reviews__ratingLabel--24XY2"})
        label.append(lbl.text)
    except:
        label.append("NA")
    
    #scrape the walk grade
    try:
        wlk = soup.find("span", attrs={"class":"hotels-hotel-review-location-layout-Highlight__number--S3wsZ hotels-hotel-review-location-layout-Highlight__green--3lccI"})
        walk.append(wlk.text)
    except:
        walk.append("NA")
    
    #scrape the no. of restaurants
    try:
        resr = soup.find("span", attrs={"class":"hotels-hotel-review-location-layout-Highlight__number--S3wsZ hotels-hotel-review-location-layout-Highlight__orange--1N-BP"})
        restaurant.append(resr.text)
    except:
        restaurant.append("NA")
    
    #scrape the no. of attractions
    try:
        attr = soup.find("span", attrs={"class":"hotels-hotel-review-location-layout-Highlight__number--S3wsZ hotels-hotel-review-location-layout-Highlight__blue--2qc3K"})
        attraction.append(attr.text)
    except:
        attraction.append("NA")
        
    #scrape the state code
    try:
        st = []
        for item in soup.find_all("li" , attrs={"class":"breadcrumb"}): #this will scrape the country, location,...
            st.append(item.text)
        st1 = st[1] # the 2nd item will contain the state name and its code
        st2=re.search('\(([^)]+)', st1).group(1) #get the word within the first parenthesis
        state.append(st2)
    except:
        state.append("NA")
   
    
```

# Save the scrapped data in a CSV file 


```python
df1 = pd.DataFrame(name).T
df2 = pd.DataFrame(location).T
df3 = pd.DataFrame(price).T
df4 = pd.DataFrame(score).T
df5 = pd.DataFrame(label).T
df6 = pd.DataFrame(walk).T
df7 = pd.DataFrame(restaurant).T
df8 = pd.DataFrame(attraction).T
df9 = pd.DataFrame(state).T
df = pd.concat((df1,df2,df9,df3,df4,df5,df6,df7,df8)).T # merge the individual data into a dataframe
df.columns = ("Hotel","Location","Code","Cost","Score","Rating","Walk.Grade","No. Restaurants","No. Attractions")
df.drop_duplicates(inplace=True) # drop duplicates, fortunately, I have no duplicates after the scraping


```


```python
df # show my dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hotel</th>
      <th>Location</th>
      <th>Code</th>
      <th>Cost</th>
      <th>Score</th>
      <th>Rating</th>
      <th>Walk.Grade</th>
      <th>No. Restaurants</th>
      <th>No. Attractions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Baccarat Hotel &amp; Residences New York</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$1,045</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100</td>
      <td>451</td>
      <td>119</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Crowne Plaza Times Square Manhattan</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$229</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100</td>
      <td>551</td>
      <td>246</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Park Lane Hotel</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$180</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100</td>
      <td>263</td>
      <td>90</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Martinique New York on Broadway, Curio Collect...</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$191</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100</td>
      <td>547</td>
      <td>104</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Arlo NoMad</td>
      <td>NA</td>
      <td>NY</td>
      <td>$215</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100</td>
      <td>511</td>
      <td>89</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>12308</td>
      <td>Beach Walk Oceanfront Inn</td>
      <td>Old Orchard Beach</td>
      <td>ME</td>
      <td>NA</td>
      <td>3.5</td>
      <td>Very good</td>
      <td>71</td>
      <td>50</td>
      <td>9</td>
    </tr>
    <tr>
      <td>12309</td>
      <td>Moby Dick Motel</td>
      <td>Old Orchard Beach</td>
      <td>ME</td>
      <td>NA</td>
      <td>3.5</td>
      <td>Very good</td>
      <td>87</td>
      <td>57</td>
      <td>12</td>
    </tr>
    <tr>
      <td>12310</td>
      <td>Sir Charles Motel</td>
      <td>Old Orchard Beach</td>
      <td>ME</td>
      <td>NA</td>
      <td>3.5</td>
      <td>Very good</td>
      <td>83</td>
      <td>56</td>
      <td>11</td>
    </tr>
    <tr>
      <td>12311</td>
      <td>Sunset Motel</td>
      <td>Old Orchard Beach</td>
      <td>ME</td>
      <td>NA</td>
      <td>3.0</td>
      <td>Average</td>
      <td>100</td>
      <td>47</td>
      <td>11</td>
    </tr>
    <tr>
      <td>12312</td>
      <td>Seafarer Inn and Cottages</td>
      <td>Old Orchard Beach</td>
      <td>ME</td>
      <td>$125</td>
      <td>2.0</td>
      <td>Poor</td>
      <td>62</td>
      <td>12</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>12313 rows × 9 columns</p>
</div>




```python
df.to_csv("TripAd-U.S_Hotels.csv", sep='\t') # save it in a CSV file
```

```python
from pandas import DataFrame
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from functools import reduce
import plotly.graph_objects as go



```

# Part I. Data preprocessing

### Read CSV which was saved from scraping process


```python
hotel = pd.read_csv('TripAd-U.S_Hotels.csv', '\t', index_col=0) # read csv file and save it in hotel
```

### Overview of the dataset


```python
length = len(hotel)
hotel.head(10)
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
      <td>100.0</td>
      <td>451.0</td>
      <td>119.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Crowne Plaza Times Square Manhattan</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$229</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>551.0</td>
      <td>246.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Park Lane Hotel</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$180</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>263.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Martinique New York on Broadway, Curio Collect...</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$191</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>547.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Arlo NoMad</td>
      <td>NaN</td>
      <td>NY</td>
      <td>$215</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>511.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Hilton Garden Inn New York Times Square South</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$280</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>406.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Arlo SoHo</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$179</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>162.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>The Gotham Hotel</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$215</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>452.0</td>
      <td>141.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>The Lexington Hotel, Autograph Collection</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$189</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>482.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Hilton Times Square</td>
      <td>New York City</td>
      <td>NY</td>
      <td>$215</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>597.0</td>
      <td>236.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel.dtypes  #check data types
```




    Hotel               object
    Location            object
    Code                object
    Cost                object
    Score              float64
    Rating              object
    Walk.Grade         float64
    No. Restaurants    float64
    No. Attractions    float64
    dtype: object



### There are two main things to preprocess with the data:

There are some missing values in the location. I will reasign the name of the location for those missing values.<br>
Bacsically, my hotels was grouped by location when I scrape them so I can asign the missing location for a hotel by naming the location as the hotel above/below. <br><br>
The "Cost" column is not float type. Its values contain dollar sign and comma (e.g. $1,234). I need to remove the dollar signs and the commas and convert the values in "Cost" from object to float.





```python
#need to change the "Cost" column data type to float
hotel["Cost"] = hotel["Cost"].str.replace(',', '') #remove comma
hotel["Cost"] = hotel["Cost"].str.replace('$', '') #remove dollar sign
hotel["Cost"] = hotel["Cost"].fillna("0").astype(int) #convert NaN to 0 and change "Cost" to float type
hotel["Cost"].replace(0, np.nan, inplace=True) #reasign NaN into "Cost"
```


```python
hotel.dtypes #see data types now
```




    Hotel               object
    Location            object
    Code                object
    Cost               float64
    Score              float64
    Rating              object
    Walk.Grade         float64
    No. Restaurants    float64
    No. Attractions    float64
    dtype: object




```python
hotel.head(10)
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
      <td>1045.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>451.0</td>
      <td>119.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Crowne Plaza Times Square Manhattan</td>
      <td>New York City</td>
      <td>NY</td>
      <td>229.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>551.0</td>
      <td>246.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Park Lane Hotel</td>
      <td>New York City</td>
      <td>NY</td>
      <td>180.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>263.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Martinique New York on Broadway, Curio Collect...</td>
      <td>New York City</td>
      <td>NY</td>
      <td>191.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>547.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Arlo NoMad</td>
      <td>NaN</td>
      <td>NY</td>
      <td>215.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>511.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Hilton Garden Inn New York Times Square South</td>
      <td>New York City</td>
      <td>NY</td>
      <td>280.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>406.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Arlo SoHo</td>
      <td>New York City</td>
      <td>NY</td>
      <td>179.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>162.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>The Gotham Hotel</td>
      <td>New York City</td>
      <td>NY</td>
      <td>215.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>452.0</td>
      <td>141.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>The Lexington Hotel, Autograph Collection</td>
      <td>New York City</td>
      <td>NY</td>
      <td>189.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>482.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Hilton Times Square</td>
      <td>New York City</td>
      <td>NY</td>
      <td>215.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>597.0</td>
      <td>236.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#there are some NaN in location. I need to assign the missing locations.
#Basically, my dataset is arranged by location. 
#Therefore, I can know the missing location by looking at the above/below row.
#I can set the missing locations by naming them as their above/below rows.

for i in range (1,length):
    try:
        np.isnan(hotel.ix[i,"Location"]) #try if the row i contains missing location
        hotel.ix[i,"Location"] = hotel.ix[i-1,"Location"] #if True, name it as the above row.
    except:
        pass
    
    
   
```


```python
hotel.head(10)
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
      <td>1045.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>451.0</td>
      <td>119.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Crowne Plaza Times Square Manhattan</td>
      <td>New York City</td>
      <td>NY</td>
      <td>229.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>551.0</td>
      <td>246.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Park Lane Hotel</td>
      <td>New York City</td>
      <td>NY</td>
      <td>180.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>263.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Martinique New York on Broadway, Curio Collect...</td>
      <td>New York City</td>
      <td>NY</td>
      <td>191.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>547.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Arlo NoMad</td>
      <td>New York City</td>
      <td>NY</td>
      <td>215.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>511.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Hilton Garden Inn New York Times Square South</td>
      <td>New York City</td>
      <td>NY</td>
      <td>280.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>406.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Arlo SoHo</td>
      <td>New York City</td>
      <td>NY</td>
      <td>179.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>162.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>The Gotham Hotel</td>
      <td>New York City</td>
      <td>NY</td>
      <td>215.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>452.0</td>
      <td>141.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>The Lexington Hotel, Autograph Collection</td>
      <td>New York City</td>
      <td>NY</td>
      <td>189.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>100.0</td>
      <td>482.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Hilton Times Square</td>
      <td>New York City</td>
      <td>NY</td>
      <td>215.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>100.0</td>
      <td>597.0</td>
      <td>236.0</td>
    </tr>
  </tbody>
</table>
</div>



### Strange Values

There is a hotel with strangle price per night. It costs more than $200,000 per night. At first, I checked the hotel name and look it in TripAdvisor and there is nothing wront with the price. However, one day later I saw that they update the price. Therefore, I changed the price according to their updates because the previous price is extremely significant.


```python
hotel.boxplot("Cost") # There is one hotel with more than $200,000
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e3ed050>




![png](output_16_1.png)



```python
hotel[hotel.Cost == hotel["Cost"].max()] #this is the hotel that has a strange price. 
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
      <td>895</td>
      <td>Hotel Croydon</td>
      <td>Miami Beach</td>
      <td>FL</td>
      <td>227772.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>79.0</td>
      <td>22.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#The next day, I looked the TripAdvisor website, they updated the price to $149. One day after, it became $80.
#I will edit the price in my dataset to $80.

hotel.loc[hotel.Cost == hotel["Cost"].max(), "Cost"] = 80 #update the price to 80

hotel[hotel.Hotel == "Hotel Croydon"] #check the hotel 




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
      <td>895</td>
      <td>Hotel Croydon</td>
      <td>Miami Beach</td>
      <td>FL</td>
      <td>80.0</td>
      <td>4.5</td>
      <td>Excellent</td>
      <td>79.0</td>
      <td>22.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel.boxplot("Cost") #there is one hotel with nearly $7,000 per night.
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1da5ebd0>




![png](output_19_1.png)



```python
hotel[hotel.Cost == hotel["Cost"].max()]

#I checked the hotel and the price has not been changed a lot in 5 days. Seems that this price has nothing wrong.
#The link for this hotel:
#https://www.tripadvisor.com/Hotel_Review-g34543-d14969413-Reviews-Origin_at_Seahaven-Panama_City_Beach_Florida.html
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
      <td>7690</td>
      <td>Origin at Seahaven</td>
      <td>Panama City Beach</td>
      <td>FL</td>
      <td>6749.0</td>
      <td>4.0</td>
      <td>Very good</td>
      <td>62.0</td>
      <td>26.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# For the next parts, I do some preparations below
```


```python
#calculate cost average by locations and save them into a dataframe
cost_avg = hotel.groupby("Location", as_index=True)["Cost"].mean()
cost_avg = pd.DataFrame({"Location":cost_avg.index, "Cost":cost_avg.values})

#calculate score average by locations and save them into a dataframe
score_avg = hotel.groupby("Location", as_index=True)["Score"].mean()
score_avg = pd.DataFrame({"Location":score_avg.index, "Score":score_avg.values})

#calculate walk.grade average by locations and save them into a dataframe
wlk_avg = hotel.groupby("Location", as_index=True)["Walk.Grade"].mean()
wlk_avg = pd.DataFrame({"Location":wlk_avg.index, "Walk.Grade":wlk_avg.values})

#calculate cost average by locations and save them into a dataframe
rest_avg = hotel.groupby("Location", as_index=True)["No. Restaurants"].mean()
rest_avg = pd.DataFrame({"Location":rest_avg.index, "No. Restaurants":rest_avg.values})

#calculate cost average by locations and save them into a dataframe
atr_avg = hotel.groupby("Location", as_index=True)["No. Attractions"].mean()
atr_avg = pd.DataFrame({"Location":atr_avg.index, "No. Attractions":atr_avg.values})

```


```python
#calculate cost average by states and save them into a dataframe
cost_avg0 = hotel.groupby("Code", as_index=True)["Cost"].mean()
cost_avg0 = pd.DataFrame({"Code":cost_avg0.index, "Cost":cost_avg0.values})

#calculate score average by states and save them into a dataframe
score_avg0= hotel.groupby("Code", as_index=True)["Score"].mean()
score_avg0 = pd.DataFrame({"Code":score_avg0.index, "Score":score_avg0.values})

#calculate walk.grade average by states and save them into a dataframe
wlk_avg0 = hotel.groupby("Code", as_index=True)["Walk.Grade"].mean()
wlk_avg0 = pd.DataFrame({"Code":wlk_avg0.index, "Walk.Grade":wlk_avg0.values})

#calculate cost average by states and save them into a dataframe
rest_avg0 = hotel.groupby("Code", as_index=True)["No. Restaurants"].mean()
rest_avg0= pd.DataFrame({"Code":rest_avg0.index, "No. Restaurants":rest_avg0.values})

#calculate cost average by states and save them into a dataframe
atr_avg0 = hotel.groupby("Code", as_index=True)["No. Attractions"].mean()
atr_avg0 = pd.DataFrame({"Code":atr_avg0.index, "No. Attractions":atr_avg0.values})
```


```python
#merge all the dataframes above by Location
avg_list = [cost_avg, score_avg, wlk_avg, rest_avg, atr_avg] 
avg = reduce(lambda left,right: pd.merge(left,right,on="Location"), avg_list)
#avg.head(10)
```


```python
#merge all the dataframes above by Code
avg_list0 = [cost_avg0, score_avg0, wlk_avg0, rest_avg0, atr_avg0] 
avg0 = reduce(lambda left,right: pd.merge(left,right,on="Code"), avg_list0)
#avg0.head(10)
```

# Part II. Overview

## 1. U.S Hotel Quality


```python
hotel["Score"].describe() #describe Rating
```




    count    11582.000000
    mean         3.912666
    std          0.797842
    min          0.000000
    25%          3.500000
    50%          4.000000
    75%          4.500000
    max          5.000000
    Name: Score, dtype: float64



The average score is around 3.93, pretty high number. It shows that most of U.S hotels reach the Very good quality


```python
count_rating = hotel.groupby("Rating").size()
count_rating.sort_values(ascending = False, inplace = True)
```


```python
count_rating.plot.pie(shadow = True, 
    autopct = '%1.1f%%', pctdistance = 0.8, labels = None, radius = 1, 
    wedgeprops={"edgecolor":"grey",'linewidth': 1, 'antialiased': True}, figsize = (8,8))

plt.title('Rating distribution', fontsize = 'x-large')
plt.xlabel('')
plt.ylabel('')
plt.legend(title = 'Rating', labels = count_rating.index, loc = 'best', fontsize = 'medium')

```




    <matplotlib.legend.Legend at 0x1a26a2f050>




![png](output_31_1.png)


As we see from the pie chart, the majority of U.S hotels are rated Very good and Excellent.


```python
#plot top 20 locations and top 20 states by rating

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,5))
ax2.yaxis.tick_right()
fig.subplots_adjust(wspace = 0.1)
fig.suptitle("Rating", fontsize = "xx-large")

score_avg.sort_values(by="Score",ascending=True, inplace=True)
score_avg.dropna(inplace=True)
ax1.barh(data=score_avg, y=score_avg[-10:].Location, width=score_avg[-10:]["Score"], color ="darkorange")
ax1.set_title("Top 10 Locations", size="x-large")


score_avg0.sort_values(by="Score",ascending=True, inplace=True)
score_avg0.dropna(inplace=True)
ax2.barh(y=score_avg0[-10:].Code, width=score_avg0[-10:]["Score"], color ="gold")
ax2.set_title("Top 10 States", size="x-large")
```




    Text(0.5, 1.0, 'Top 10 States')




![png](output_33_1.png)


## 2. Price


```python
hotel["Cost"].describe() #describe Price

```




    count    8588.000000
    mean      140.785748
    std       132.011559
    min        20.000000
    25%        79.000000
    50%       110.000000
    75%       165.000000
    max      6749.000000
    Name: Cost, dtype: float64



The average price is nearly $141, which is a affordable for most visitors to the United States.


```python
cost_avg.dropna(inplace=True)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,5))
ax2.yaxis.tick_right()
fig.subplots_adjust(wspace = 0.1)
fig.suptitle("Price by location", fontsize ="xx-large")

cost_avg.sort_values(by="Cost",ascending=True, inplace=True) #locations with highest average price
ax1.barh(y=cost_avg[-10:].Location, width=cost_avg[-10:]["Cost"], color ="lightseagreen")
ax1.set_title("Highest Average Price", size ='x-large')

cost_avg.sort_values(by="Cost",ascending=False, inplace=True) #locations with lowest average price
ax2.barh(y=cost_avg[-10:].Location, width=cost_avg[-10:]["Cost"], color ="mediumturquoise")
ax2.set_title("Lowest Average Price", size = 'x-large')

```




    Text(0.5, 1.0, 'Lowest Average Price')




![png](output_37_1.png)



```python
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,5))
ax2.yaxis.tick_right()
fig.subplots_adjust(wspace = 0.1)
fig.suptitle("Price by State", fontsize = "xx-large")

cost_avg0.sort_values(by="Cost",ascending=True, inplace=True) #states with higest price
ax1.barh(y=cost_avg0[-10:].Code, width=cost_avg0[-10:]["Cost"], color ="lightseagreen")
ax1.set_title("Highest Average Price", size ='x-large')

cost_avg0.sort_values(by="Cost",ascending=False, inplace=True) #states with lowest price
ax2.barh(y=cost_avg0[-10:].Code, width=cost_avg0[-10:]["Cost"], color ="mediumturquoise")
ax2.set_title("Lowest Average Price", size = 'x-large')

```




    Text(0.5, 1.0, 'Lowest Average Price')




![png](output_38_1.png)


## 3. Price and Rating

### Categories of price per night

Since we do not have any specific price range, I will divide it by my own criteria after reviewing some hotel websites about the price.

* Budget: <= 150 <br>
* Moderate: more than 150 - 350 <br>
* Expensive: more than 350 - 500 <br>
* Very expensive: more than 500 - less than 1000 <br>
* Luxury: 1000+.
<br>

P/s: All prices are U.S dollars.


```python
#Now, I will subset the hotel into different groups of price
hotel_budget = hotel[hotel.Cost <= 150]
hotel_moderate = hotel[(hotel.Cost > 150) & (hotel.Cost <= 350)]
hotel_expensive = hotel[(hotel.Cost > 350) & (hotel.Cost <= 500)]
hotel_very_expensive = hotel[(hotel.Cost > 500) & (hotel.Cost < 1000)]
hotel_luxury = hotel[hotel.Cost >= 1000]
```


```python
#set the order of Rating (Excellent, Very good, ...) for better look at boxplots
order = hotel["Rating"].drop_duplicates()
order.index=range(len(order))
order.dropna(inplace=True)

```


```python
plt.figure(figsize=(7,7))
sns.boxplot(data=hotel_budget,x="Rating", y = "Cost", order=order)
plt.title("Cost distribution/ Budget", fontsize ="xx-large")
plt.xlabel("Rating", fontsize ="x-large")
plt.ylabel("Cost ($)", fontsize = "x-large")
plt.suptitle("")

```


![png](output_44_0.png)



```python
plt.figure(figsize=(7,7))
sns.boxplot(data=hotel_moderate,x="Rating", y = "Cost",order=order)
plt.title("Cost distribution/ Moderate", fontsize ="xx-large")
plt.xlabel("Rating", fontsize ="x-large")
plt.ylabel("Cost ($)", fontsize = "x-large")
plt.suptitle("")

```


![png](output_45_0.png)



```python
plt.figure(figsize=(7,7))
sns.boxplot(data=hotel_expensive,x="Rating", y = "Cost",order=order)
plt.title("Cost distribution/ Expensive", fontsize ="xx-large")
plt.xlabel("Rating", fontsize ="x-large")
plt.ylabel("Cost ($)", fontsize = "x-large")
plt.suptitle("")


```


![png](output_46_0.png)



```python
plt.figure(figsize=(7,7))
sns.boxplot(data=hotel_very_expensive,x="Rating", y = "Cost",order=order)
plt.title("Cost distribution/ Very expensive", fontsize ="xx-large")
plt.xlabel("Rating", fontsize ="x-large")
plt.ylabel("Cost ($)", fontsize = "x-large")
plt.suptitle("")


```


![png](output_47_0.png)



```python
plt.figure(figsize=(7,7))
sns.boxplot(data=hotel_luxury,x="Rating", y = "Cost",order=order)
plt.title("Cost distribution/ Luxury", fontsize ="xx-large")
plt.xlabel("Rating", fontsize ="x-large")
plt.ylabel("Cost ($)", fontsize = "x-large")
plt.suptitle("")

```


![png](output_48_0.png)


### Findings: 

* In all price ranges, excellent hotels tend to concentrate on the lower price side (the boxplots show a right-skew distribution).
* In general, when the price range increases, the rating of the hotels is better. We see terrible hotels in budget range, but in moderate range, there is no terrible hotels. In the expensive and very expensive ranges, there are only 1-2 poor hotels; and in the luxury range, all the hotels are rated excellent or very good.


## 4. Walk Grade

Walk Grade show how convenient for visitors to move to attractions, restaurants, shops,... nearby the hotel.


```python
plt.figure(figsize=(7,7))
sns.boxplot(data=hotel,x="Rating", y = "Walk.Grade",order=order)
plt.title("Walking grade distribution", fontsize ="xx-large")
plt.xlabel("Rating", fontsize ="x-large")
plt.ylabel("Grade", fontsize = "x-large")
plt.suptitle("")


```


![png](output_52_0.png)


Overall, excellent hotels tend to get very high grade for walking.

# 4. Nearby services 

## Walking


```python
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,7))
ax2.yaxis.tick_right()
fig.subplots_adjust(wspace = 0.1)
fig.suptitle("Walk Grade", fontsize = "xx-large")

wlk_avg.sort_values(by="Walk.Grade",ascending=True, inplace=True)
wlk_avg.dropna(inplace=True)
ax1.barh(data=wlk_avg, y=wlk_avg[-20:].Location, width=wlk_avg[-20:]["Walk.Grade"], color ="royalblue")
ax1.set_title("Location", size="x-large")


wlk_avg0.sort_values(by="Walk.Grade",ascending=True, inplace=True)
wlk_avg0.dropna(inplace=True)
ax2.barh(y=wlk_avg0[-20:].Code, width=wlk_avg0[-20:]["Walk.Grade"], color ="dodgerblue")
ax2.set_title("State", size="x-large")
```




    Text(0.5, 1.0, 'State')




![png](output_56_1.png)


## Restaurant


```python
#top 20 locations and top 20 states by the average number of restaurants nearby a hotel

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,7))
ax2.yaxis.tick_right()
fig.subplots_adjust(wspace = 0.1)
fig.suptitle("Number of Restaurants", fontsize = "xx-large")

rest_avg.sort_values(by="No. Restaurants",ascending=True, inplace=True)
rest_avg.dropna(inplace=True)
ax1.barh(y=rest_avg[-20:].Location, width=rest_avg[-20:]["No. Restaurants"], color ="yellowgreen")
ax1.set_title("Location", size="x-large")

rest_avg0.sort_values(by="No. Restaurants",ascending=True, inplace=True)
rest_avg0.dropna(inplace=True)
ax2.barh(y=rest_avg0[-20:].Code, width=rest_avg0[-20:]["No. Restaurants"], color ="lightgreen")
ax2.set_title("State", size="x-large")
```




    Text(0.5, 1.0, 'State')




![png](output_58_1.png)


## Attraction


```python
#top 20 locations and top 20 states by the average number of attractions nearby a hotel

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,7))
ax2.yaxis.tick_right()
fig.subplots_adjust(wspace = 0.1)
fig.suptitle("Number of Attractions", fontsize = "xx-large")

atr_avg.sort_values(by="No. Attractions",ascending=True, inplace=True)
atr_avg.dropna(inplace=True)
ax1.barh(y=atr_avg[-20:].Location, width=atr_avg[-20:]["No. Attractions"], color ="mediumvioletred")
ax1.set_title("Location", size="x-large")

atr_avg0.sort_values(by="No. Attractions",ascending=True, inplace=True)
atr_avg0.dropna(inplace=True)
ax2.barh(y=atr_avg0[-20:].Code, width=atr_avg0[-20:]["No. Attractions"], color ="deeppink")
ax2.set_title("State", size="x-large")
```




    Text(0.5, 1.0, 'State')




![png](output_60_1.png)


### Findings:
* D.C has the highest grade for walking.
* New York City has the highest average number of restaurants and attractions neary a hotel. The city's state NY is also at the top of the state list.

# Part III. Analysis

# 1. Best Qualtity Hotels & Best Price Hotels

* Best quality hotels contain all hotels rated Excellent and Very good.
* Best price hotels contain all hotels with price per night not more than $350.


```python
best_hotel = hotel[(hotel.Rating == "Excellent") | (hotel.Rating == "Very good")] #best quality
reasonable = pd.concat((hotel_budget,hotel_moderate)) #best price

```


```python
count1 = best_hotel.groupby("Location").size()
count1 = pd.DataFrame({"Location":count1.index, "No. Hotels":count1.values})
count1.sort_values(by="No. Hotels",ascending=True, inplace=True)

count2 = reasonable.groupby("Location").size()
count2 = pd.DataFrame({"Location":count2.index, "No. Hotels":count2.values})
count2.sort_values(by="No. Hotels",ascending=True, inplace=True)

count10 = best_hotel.groupby("Code").size()
count10 = pd.DataFrame({"Code":count10.index, "No. Hotels":count10.values})
count10.sort_values(by="No. Hotels",ascending=True, inplace=True)

count20 = reasonable.groupby("Code").size()
count20 = pd.DataFrame({"Code":count20.index, "No. Hotels":count20.values})
count20.sort_values(by="No. Hotels",ascending=True, inplace=True)

```

### Locations with most best quality and best price hotels


```python
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,7))
ax2.yaxis.tick_right()
fig.subplots_adjust(wspace = 0.1)
ax1.barh(y=count1[-20:].Location, width=count1[-20:]["No. Hotels"], color ="lightskyblue")
ax1.set_title("20 Locations with most good hotels", size ='xx-large')
ax2.barh(y=count2[-20:].Location, width=count2[-20:]["No. Hotels"], color ="bisque")
ax2.set_title("20 Locations with most affordable hotels", size = 'xx-large')



```




    Text(0.5, 1.0, '20 Locations with most affordable hotels')




![png](output_68_1.png)


### States with most best quality and best price hotels


```python
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,7))
ax2.yaxis.tick_right()
fig.subplots_adjust(wspace = 0.1)
ax1.barh(y=count10[-20:].Code, width=count10[-20:]["No. Hotels"], color ="royalblue")
ax1.set_title("20 States with most good hotels", size ='xx-large')
ax2.barh(y=count20[-20:].Code, width=count20[-20:]["No. Hotels"], color ="coral")
ax2.set_title("20 States with most affordable hotels", size = 'xx-large')


```


![png](output_70_0.png)



```python
#define a function to help to plot faster. I should do this at the start, but I realize that I need it too late.

def barhplot(df1,df2,group,name):
    set1 = df1.groupby(group, as_index=True)[name].mean()
    set1 = pd.DataFrame({group:set1.index, name:set1.values})
    
    set2 = df2.groupby(group, as_index=True)[name].mean()
    set2 = pd.DataFrame({group:set2.index, name:set2.values})
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,5))
    ax2.yaxis.tick_right()
    fig.subplots_adjust(wspace = 0.1)
    fig.suptitle(name, fontsize ='xx-large')

    set1.sort_values(by=name,ascending=True, inplace=True)
    set1.dropna(inplace=True)
    ax1.barh(y=set1[-10:][group], width=set1[-10:][name], color ="dodgerblue")
    ax1.set_title("Good Rating", size = "x-large")

    set2.sort_values(by=name,ascending=True, inplace=True)
    set2.dropna(inplace=True)
    ax2.barh(y=set2[-10:][group], width=set2[-10:][name], color ="tomato")
    ax2.set_title("Reasonable Price", size = "x-large")

```

## Nearby Services 

### Location


```python
barhplot(best_hotel,reasonable,"Location","Walk.Grade")
```


![png](output_74_0.png)



```python
barhplot(best_hotel,reasonable,"Location","No. Restaurants")
```


![png](output_75_0.png)



```python
barhplot(best_hotel,reasonable,"Location","No. Attractions")
```


![png](output_76_0.png)


### State


```python
barhplot(best_hotel,reasonable,"Code","Walk.Grade")
```


![png](output_78_0.png)



```python
barhplot(best_hotel,reasonable,"Code","No. Restaurants")
```


![png](output_79_0.png)



```python
barhplot(best_hotel,reasonable,"Code","No. Attractions")
```


![png](output_80_0.png)


# 2. Best Value Hotels

I define best value hotel by rating and price per night:
* Hotel has budget or moderate price: price per night is not more than 350 U.S dollars
* Hotel was rated Excellent or Very good.

In summary, best value hotel = best price hotel + best quality hotel.


```python
#Here is my best value hotel dataframe
best_value = hotel[(hotel.Cost <= 350) & ((hotel.Rating == "Very good") | (hotel.Rating == "Excellent"))]
```


```python
# Some pre-calculations before analysis. 
# Information about price and rating and score will be excluded since they are not necessary for next analysis.
# I've already select the best price and best quality hotels for my next analysis.
```


```python
state_count = best_value.groupby("Code").size() #count number of best value hotels in each state
state_count = pd.DataFrame({"Code":state_count.index, "No. Hotels":state_count.values})
state_count.sort_values(by="No. Hotels",ascending=True, inplace=True)
top = state_count["Code"][-20:].tolist() #save codes of top 20 states into a list


wlkmean = best_value.groupby("Code", as_index=True)["Walk.Grade"].mean() #walk grade average by state
wlkmean = pd.DataFrame({"Code":wlkmean.index, "Walk.Grade":wlkmean.values})


restmean = best_value.groupby("Code", as_index=True)["No. Restaurants"].mean() #average no. of restaurants
restmean = pd.DataFrame({"Code":restmean.index, "No. Restaurants":restmean.values})


atrmean = best_value.groupby("Code", as_index=True)["No. Attractions"].mean() #average no. of attractions
atrmean = pd.DataFrame({"Code":atrmean.index, "No. Attractions":atrmean.values})

#merge all the dataframes above by Code
mean = [wlkmean, restmean, atrmean] 
best_value_code = reduce(lambda left,right: pd.merge(left,right,on="Code"), mean)
#best_value_code.head(10)

top_state = best_value_code[best_value_code.Code.isin(top)] # save top 20 states by no. hotels into another df
```


```python
#I conduct the same process with Location

loc_count = best_value.groupby("Location").size() 
loc_count = pd.DataFrame({"Location":loc_count.index, "No. Hotels":loc_count.values})
loc_count.sort_values(by="No. Hotels",ascending=True, inplace=True)
top1 = loc_count["Location"][-20:].tolist() #save names of top 20 locations into a list


wlkmean1 = best_value.groupby("Location", as_index=True)["Walk.Grade"].mean()
wlkmean1 = pd.DataFrame({"Location":wlkmean1.index, "Walk.Grade":wlkmean1.values})


restmean1 = best_value.groupby("Location", as_index=True)["No. Restaurants"].mean()
restmean1 = pd.DataFrame({"Location":restmean1.index, "No. Restaurants":restmean1.values})


atrmean1 = best_value.groupby("Location", as_index=True)["No. Attractions"].mean()
atrmean1 = pd.DataFrame({"Location":atrmean1.index, "No. Attractions":atrmean1.values})

#merge all the dataframes above by Location
mean1 = [wlkmean1, restmean1, atrmean1] 
best_value_loc = reduce(lambda left,right: pd.merge(left,right,on="Location"), mean1)
#best_value_loc.head(10)

top_loc = best_value_loc[best_value_loc.Location.isin(top1)] # save top 20 locations by no. hotels into another df
```

## Top States


```python
plt.figure(figsize = (7,7))
plt.title("Top 20 States with most best value hotels", fontsize="x-large")
plt.barh(data=state_count, y=state_count[-20:].Code, width=count1[-20:]["No. Hotels"], color ="dodgerblue")
```




    <BarContainer object of 20 artists>




![png](output_88_1.png)



```python
top_state.plot(figsize = (15,8), kind="bar", x="Code")
plt.xlabel("State", fontsize ="large")
plt.title("Top 20 States with the number of best value hotels", fontsize = "x-large")



```


![png](output_89_0.png)



```python
plt.figure(figsize = (7,7))
plt.title("Top 20 Locations with most best value hotels", fontsize="x-large")
plt.barh(data=loc_count, y=loc_count[-20:].Location, width=count1[-20:]["No. Hotels"], color ="orangered")
```




    <BarContainer object of 20 artists>




![png](output_90_1.png)



```python
top_loc.plot(figsize = (15,8), kind="bar", x="Location")
plt.xlabel("Destination", fontsize = "large")
plt.title("Top 20 Locations with the number of best value hotels", fontsize = "x-large")

```


![png](output_91_0.png)


# Part IV. Hotel Service Choropleth Map

I use plotly to plot the maps. I learned the codes from https://plot.ly/python/choropleth-maps/

## All hotels in the data


```python
fig = go.Figure(data=go.Choropleth(
    locations=avg0["Code"], 
    z = avg0["Score"].astype(float),  
    locationmode = 'USA-states', 
    colorscale = 'portland',
    colorbar_title = "Score",
    marker_line_color="white"))

fig.update_layout(
    title_text = 'Average Hotel Rating in the United States',
    geo_scope='usa')
fig.show()
```


<div>


            <div id="f239efd0-648b-4e36-8ad0-c6883533167e" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("f239efd0-648b-4e36-8ad0-c6883533167e")) {
                    Plotly.newPlot(
                        'f239efd0-648b-4e36-8ad0-c6883533167e',
                        [{"colorbar": {"title": {"text": "Score"}}, "colorscale": [[0.0, "rgb(12,51,131)"], [0.25, "rgb(10,136,186)"], [0.5, "rgb(242,211,56)"], [0.75, "rgb(242,143,56)"], [1.0, "rgb(217,30,30)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "DC", "FL", "GA", "HI", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MO", "NC", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VA", "WA", "WI"], "marker": {"line": {"color": "white"}}, "type": "choropleth", "z": [4.135983263598327, 3.393258426966292, 4.533898305084746, 3.9215867158671585, 3.929705215419501, 4.026819923371647, 4.1466666666666665, 3.888181818181818, 3.7189349112426036, 4.177007299270073, 4.36, 3.9083969465648853, 3.7872340425531914, 3.813953488372093, 4.234615384615385, 4.016666666666667, 3.6587677725118484, 4.129441624365482, 3.9042821158690177, 3.9684684684684686, 3.4415584415584415, 3.8923076923076922, 4.0, 3.9602122015915118, 3.7569444444444446, 3.7154471544715446, 3.7125984251968505, 3.9469026548672566, 4.242105263157895, 3.672209026128266, 3.729251700680272, 3.978448275862069, 3.936868686868687, 3.632758620689655, 3.8529411764705883, 3.7596153846153846]}],
                        {"geo": {"scope": "usa"}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average Hotel Rating in the United States"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('f239efd0-648b-4e36-8ad0-c6883533167e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true})
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



    <Figure size 432x288 with 0 Axes>


Top 5 states with highest average rating scores are Illinois (IL), Rhode Island (IR), Arkansas (AR), Los Angeles (LA) and New York (NY).


```python
fig = go.Figure(data=go.Choropleth(
    locations=avg0["Code"], 
    z = avg0["No. Restaurants"].astype(float),  
    locationmode = 'USA-states', 
    colorscale = 'greens',
    colorbar_title = "No. of Restaurants",
    marker_line_color="white"))

fig.update_layout(
    title_text = 'Average number of Restaurants nearby a hotel',
    geo_scope='usa')
fig.show()
```


<div>


            <div id="85ba2dd0-625c-43a0-afa5-b4ac7520ee57" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("85ba2dd0-625c-43a0-afa5-b4ac7520ee57")) {
                    Plotly.newPlot(
                        '85ba2dd0-625c-43a0-afa5-b4ac7520ee57',
                        [{"colorbar": {"title": {"text": "No. of Restaurants"}}, "colorscale": [[0.0, "rgb(247,252,245)"], [0.125, "rgb(229,245,224)"], [0.25, "rgb(199,233,192)"], [0.375, "rgb(161,217,155)"], [0.5, "rgb(116,196,118)"], [0.625, "rgb(65,171,93)"], [0.75, "rgb(35,139,69)"], [0.875, "rgb(0,109,44)"], [1.0, "rgb(0,68,27)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "DC", "FL", "GA", "HI", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MO", "NC", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VA", "WA", "WI"], "marker": {"line": {"color": "white"}}, "type": "choropleth", "z": [44.89510489510489, 43.431372549019606, 33.72727272727273, 51.25787965616046, 99.82207792207792, 72.18157894736842, 111.70547945205479, 64.14745884037222, 53.713740458015266, 80.91466666666666, 193.90972222222223, 54.25, 29.613636363636363, 55.58139534883721, 124.0672268907563, 115.6, 55.744897959183675, 46.074829931972786, 56.61290322580645, 69.41071428571429, 39.620253164556964, 52.92079207920792, 71.4963503649635, 234.2469512195122, 93.26582278481013, 47.10191082802548, 98.72897196261682, 125.20809248554913, 52.04123711340206, 58.46666666666667, 56.87380497131931, 60.492242595204516, 65.76, 47.373873873873876, 129.4368932038835, 30.75]}],
                        {"geo": {"scope": "usa"}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average number of Restaurants nearby a hotel"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('85ba2dd0-625c-43a0-afa5-b4ac7520ee57');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


New York (NY) has the higest number of restaurants near a hotel.<br>
Illinois (IL), Los Angeles (LA) and Washington (WA) come next in the list.


```python
fig = go.Figure(data=go.Choropleth(
    locations=avg0["Code"], 
    z = avg0["No. Attractions"].astype(float), 
    locationmode = 'USA-states', 
    colorscale = 'blues',
    colorbar_title = "No. of Attractions",
    marker_line_color="white"
))

fig.update_layout(
    title_text = 'Average number of Attractions nearby a hotel',
    geo_scope='usa')
fig.show()
```


<div>


            <div id="73926c13-26ec-4acf-ae0e-57fe05fcf2d2" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("73926c13-26ec-4acf-ae0e-57fe05fcf2d2")) {
                    Plotly.newPlot(
                        '73926c13-26ec-4acf-ae0e-57fe05fcf2d2',
                        [{"colorbar": {"title": {"text": "No. of Attractions"}}, "colorscale": [[0.0, "rgb(247,251,255)"], [0.125, "rgb(222,235,247)"], [0.25, "rgb(198,219,239)"], [0.375, "rgb(158,202,225)"], [0.5, "rgb(107,174,214)"], [0.625, "rgb(66,146,198)"], [0.75, "rgb(33,113,181)"], [0.875, "rgb(8,81,156)"], [1.0, "rgb(8,48,107)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "DC", "FL", "GA", "HI", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MO", "NC", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VA", "WA", "WI"], "marker": {"line": {"color": "white"}}, "type": "choropleth", "z": [16.237762237762237, 8.549019607843137, 36.15151515151515, 18.12320916905444, 26.637662337662338, 30.057894736842105, 17.86986301369863, 20.078740157480315, 16.02671755725191, 30.642666666666667, 44.25694444444444, 10.541666666666666, 6.204545454545454, 12.046511627906977, 73.01680672268908, 30.70909090909091, 12.025510204081632, 18.755102040816325, 30.921146953405017, 20.5, 17.936708860759495, 34.04950495049505, 33.85401459854015, 52.34756097560975, 14.746835443037975, 8.375796178343949, 19.49532710280374, 27.566473988439306, 35.63917525773196, 26.495652173913044, 30.795411089866157, 15.664315937940762, 21.48, 10.896396396396396, 26.601941747572816, 17.31578947368421]}],
                        {"geo": {"scope": "usa"}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average number of Attractions nearby a hotel"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('73926c13-26ec-4acf-ae0e-57fe05fcf2d2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


New York (NY) and Los Angeles (LA) have the highest numbers of attractions around a hotel.

## Best value hotels


```python
fig = go.Figure(data=go.Choropleth(
    locations=state_count["Code"], 
    z = state_count["No. Hotels"].astype(float), 
    locationmode = 'USA-states', 
    colorscale = 'plotly3',
    colorbar_title = "Number",
    marker_line_color="white"
))

fig.update_layout(
    title_text = "Number of Best value Hotels",
    geo_scope='usa')
fig.show()
```


<div>


            <div id="1019b6d9-de88-42da-ab9c-570fe26b0db1" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("1019b6d9-de88-42da-ab9c-570fe26b0db1")) {
                    Plotly.newPlot(
                        '1019b6d9-de88-42da-ab9c-570fe26b0db1',
                        [{"colorbar": {"title": {"text": "Number"}}, "colorscale": [[0.0, "#0508b8"], [0.08333333333333333, "#1910d8"], [0.16666666666666666, "#3c19f0"], [0.25, "#6b1cfb"], [0.3333333333333333, "#981cfd"], [0.4166666666666667, "#bf1cfd"], [0.5, "#dd2bfd"], [0.5833333333333334, "#f246fe"], [0.6666666666666666, "#fc67fd"], [0.75, "#fe88fc"], [0.8333333333333334, "#fea5fd"], [0.9166666666666666, "#febefe"], [1.0, "#fec3fe"]], "locationmode": "USA-states", "locations": ["NJ", "WI", "ME", "AR", "RI", "AK", "AL", "MA", "KS", "WA", "KY", "MD", "OR", "UT", "OH", "IN", "LA", "DC", "NV", "IL", "VA", "NM", "PA", "HI", "OK", "NY", "SC", "GA", "NC", "CO", "MO", "AZ", "TN", "CA", "TX", "FL"], "marker": {"line": {"color": "white"}}, "type": "choropleth", "z": [8.0, 27.0, 32.0, 46.0, 47.0, 55.0, 58.0, 65.0, 66.0, 78.0, 84.0, 86.0, 89.0, 92.0, 106.0, 114.0, 120.0, 136.0, 137.0, 143.0, 147.0, 154.0, 155.0, 178.0, 182.0, 193.0, 203.0, 226.0, 232.0, 256.0, 270.0, 316.0, 418.0, 672.0, 770.0, 1015.0]}],
                        {"geo": {"scope": "usa"}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Number of Best value Hotels"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('1019b6d9-de88-42da-ab9c-570fe26b0db1');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


Florida (FL), Texas (TX) and California (CA) have the higest numbers of best value hotels.


```python
fig = go.Figure(data=go.Choropleth(
    locations=best_value_code["Code"], 
    z = best_value_code["No. Restaurants"].astype(float), 
    locationmode = 'USA-states', 
    colorscale = 'greens',
    colorbar_title = "No. of Restaurants",
    marker_line_color="white"
))

fig.update_layout(
    title_text = "Average number of Restaurants/ Best value Hotels",
    geo_scope='usa')
fig.show()
```


<div>


            <div id="c12c11aa-789c-48e2-907e-869b39b66c5d" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("c12c11aa-789c-48e2-907e-869b39b66c5d")) {
                    Plotly.newPlot(
                        'c12c11aa-789c-48e2-907e-869b39b66c5d',
                        [{"colorbar": {"title": {"text": "No. of Restaurants"}}, "colorscale": [[0.0, "rgb(247,252,245)"], [0.125, "rgb(229,245,224)"], [0.25, "rgb(199,233,192)"], [0.375, "rgb(161,217,155)"], [0.5, "rgb(116,196,118)"], [0.625, "rgb(65,171,93)"], [0.75, "rgb(35,139,69)"], [0.875, "rgb(0,109,44)"], [1.0, "rgb(0,68,27)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "DC", "FL", "GA", "HI", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MO", "NC", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VA", "WA", "WI"], "marker": {"line": {"color": "white"}}, "type": "choropleth", "z": [42.48, 42.60526315789474, 31.825, 49.81651376146789, 103.0761421319797, 89.8, 112.52631578947368, 63.08361970217641, 57.27173913043478, 92.32867132867133, 194.15328467153284, 55.49230769230769, 30.580645161290324, 64.73333333333333, 124.56880733944953, 129.77966101694915, 65.45, 38.68181818181818, 54.67156862745098, 60.52795031055901, 45.75, 54.968, 71.28, 340.26704545454544, 84.3076923076923, 42.13008130081301, 115.72368421052632, 128.8655462184874, 53.06818181818182, 61.49700598802395, 52.66158536585366, 64.63265306122449, 63.03030303030303, 43.82113821138211, 126.56944444444444, 31.782608695652176]}],
                        {"geo": {"scope": "usa"}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average number of Restaurants/ Best value Hotels"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('c12c11aa-789c-48e2-907e-869b39b66c5d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
fig = go.Figure(data=go.Choropleth(
    locations=best_value_code["Code"], 
    z = best_value_code["No. Attractions"].astype(float), 
    locationmode = 'USA-states', 
    colorscale = 'blues',
    colorbar_title = "No. of Attractions",
    marker_line_color="white"
))

fig.update_layout(
    title_text = 'Average number of Attractions/ Best value Hotels',
    geo_scope='usa')
fig.show()
```


<div>


            <div id="3a04d229-34b1-45d4-be8b-9761a01c8080" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("3a04d229-34b1-45d4-be8b-9761a01c8080")) {
                    Plotly.newPlot(
                        '3a04d229-34b1-45d4-be8b-9761a01c8080',
                        [{"colorbar": {"title": {"text": "No. of Attractions"}}, "colorscale": [[0.0, "rgb(247,251,255)"], [0.125, "rgb(222,235,247)"], [0.25, "rgb(198,219,239)"], [0.375, "rgb(158,202,225)"], [0.5, "rgb(107,174,214)"], [0.625, "rgb(66,146,198)"], [0.75, "rgb(33,113,181)"], [0.875, "rgb(8,81,156)"], [1.0, "rgb(8,48,107)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "DC", "FL", "GA", "HI", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MO", "NC", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VA", "WA", "WI"], "marker": {"line": {"color": "white"}}, "type": "choropleth", "z": [13.9, 8.81578947368421, 32.425, 16.43577981651376, 28.978003384094755, 29.11111111111111, 17.646616541353385, 18.004581901489118, 16.391304347826086, 34.3986013986014, 43.941605839416056, 11.23076923076923, 6.096774193548387, 14.2, 73.37614678899082, 33.83050847457627, 15.1, 18.681818181818183, 28.61764705882353, 13.975155279503106, 20.75, 31.12, 33.616, 83.0625, 12.984615384615385, 7.390243902439025, 23.776315789473685, 27.689075630252102, 39.43181818181818, 25.37125748502994, 23.59451219512195, 15.144712430426717, 19.954545454545453, 10.829268292682928, 26.694444444444443, 16.217391304347824]}],
                        {"geo": {"scope": "usa"}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average number of Attractions/ Best value Hotels"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('3a04d229-34b1-45d4-be8b-9761a01c8080');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
fig = go.Figure(data=go.Choropleth(
    locations=avg0["Code"], 
    z = avg0["Walk.Grade"].astype(float), 
    locationmode = 'USA-states', 
    colorscale = 'fall',
    colorbar_title = "Grade",
    marker_line_color="white"
))

fig.update_layout(
    title_text = 'Average Walk Grade/ Best value Hotels',
    geo_scope='usa')
fig.show()
```


<div>


            <div id="9686008b-7563-4115-af72-8a6a600f00b1" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("9686008b-7563-4115-af72-8a6a600f00b1")) {
                    Plotly.newPlot(
                        '9686008b-7563-4115-af72-8a6a600f00b1',
                        [{"colorbar": {"title": {"text": "Grade"}}, "colorscale": [[0.0, "rgb(61, 89, 65)"], [0.16666666666666666, "rgb(119, 136, 104)"], [0.3333333333333333, "rgb(181, 185, 145)"], [0.5, "rgb(246, 237, 189)"], [0.6666666666666666, "rgb(237, 187, 138)"], [0.8333333333333334, "rgb(222, 138, 90)"], [1.0, "rgb(202, 86, 44)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "DC", "FL", "GA", "HI", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MO", "NC", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VA", "WA", "WI"], "marker": {"line": {"color": "white"}}, "type": "choropleth", "z": [68.37762237762237, 61.627450980392155, 73.57575757575758, 63.69914040114613, 76.22727272727273, 72.40526315789474, 95.88356164383562, 70.85254115962778, 67.35114503816794, 73.27466666666666, 92.95833333333333, 68.59722222222223, 53.59090909090909, 61.19767441860465, 93.83193277310924, 90.13636363636364, 84.76020408163265, 78.4421768707483, 67.15053763440861, 60.875, 92.45569620253164, 66.97029702970298, 80.87591240875912, 86.16158536585365, 64.0379746835443, 59.63694267515923, 72.32710280373831, 83.35838150289017, 85.50515463917526, 68.98840579710145, 70.30210325047801, 66.08603667136812, 73.68, 71.34684684684684, 88.58252427184466, 71.23684210526316]}],
                        {"geo": {"scope": "usa"}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average Walk Grade/ Best value Hotels"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('9686008b-7563-4115-af72-8a6a600f00b1');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


For the best value hotels data, the state with highest average numbers of restaurants and attractions around a hotel is New York (NY). The result is the same when considering all hotels data.<br>
New York (NY), Los Angeles (LA) and Illinois (IL) seem to be the best places if visitors want to wander and explore the stores, restaurants, attractions ... in the United States.

# Part V. Conclusion

My project only reflects a part of U.S hotel industry. The dataset is scraped from around 12,300 hotels in the top 100 popular locations in the United States. In addition, the results show the places with the most sightseeing options (restaurants, attractions), but can not point out the best destination to visit.<br>

Overall, the hotel industry in the United States is growing very well. The majority of hotels (around 83%) in the country are rated as Excellent or Very good. The price is also reasonable. It costs average 140 USD per night to stay at a hotel.
<br>
New York, Los Angeles and Illinois seem to have the various options for their tourists. They have a lot of restaurants, attractions nearby a hotel. Furthermore, the prices to stay are affordable since there are many best value hotels in these states.   
<br>
If visitors love walking around the hotel and experience the shopping malls and places of interest, New York, Los Angeles and Illinois are also perfect selections. These states have very high grades in walking. 

<br>
It is noteworthy that D.C has the higest grade for walking and is among the top states of best value hotels. Washington D.C also has a high walking grade and decent number of best value hotels. Although there are modest numbers of restaurants and attractions, it is worth to consider visiting D.C.





```python

```

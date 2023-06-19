import streamlit as st
import pandas as pd
import datetime
from datetime import datetime as dt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data2 = pd.read_csv('acnh_villagers.csv')

st.title('Make-Your-Own-Villagers Recommendation')
random_rows = data2.sample(n=1)
st.caption(random_rows['nh_details.quote'].values[0] + ' - ' + random_rows['Name'].values[0])

with st.sidebar:
    st.image('https://dodo.ac/np/images/5/52/NH_Logo_English.png', use_column_width='auto')
    st.audio('https://dodo.ac/np/images/4/48/NH_Main_Theme.flac', format='audio/flac', start_time=0)
    st.write('---')

st.write('---')
st.write('Select the filters that you wish to apply:')
filtcol1, filtcol2 = st.columns(2)
with filtcol1:
    filter1 = st.selectbox('Species filter', 
                        [None] + list(data2['Species'].unique()))
with filtcol2:
    filter2 = st.selectbox('Gender filter',
        [None, "Male", "Female"])
    
# data multiple filter
if filter1 is not None:
    data2 = data2[data2['Species'] == filter1]
if filter2 is not None:
    data2 = data2[data2['Gender'] == filter2]
    
data2['PersonalitySub'] = data2['Personality'] + data2['nh_details.sub-personality']
data2['Birthday'] = data2['birthday_month'].astype(str) + ' ' + data2['birthday_day'].astype(str)

main_out = ['Name', 'Species', 'Gender', 'Hobby', 'Birthday', 
            'Personality', 'Catchphrase', 'Favorite Song',
            'Style 1', 'Style 2', 'Color 1', 'Color 2']

main = ['Name', 'Species', 'Gender', 'Hobby', 'PersonalitySub', 
            'Catchphrase', 'Favorite Song', 'Birthday',
            'Style 1', 'Style 2', 'Color 1', 'Color 2', 
            'birthday_month', 'sign', 'nh_details.quote']

features = ['Species', 'Gender', 'Hobby', 'PersonalitySub', 
            'Catchphrase', 'Favorite Song', 'Birthday',
            'Style 1', 'Style 2', 'Color 1', 'Color 2', 
            'birthday_month', 'sign', 'nh_details.quote']

df2 = data2[main]

st.write('---')
optcol1, optcol2 = st.columns(2)

with optcol1:
    option1 = st.selectbox('Select your species', 
                        data2['Species'].unique())

    option2 = st.selectbox('Select your gender',
        ["Male", "Female", "I'd rather not say"])

    option3 = st.selectbox('Select your hobby',
                        data2['Hobby'].unique())

    option4 = st.text_input('Insert a catchphrase', 'hi there')

    option5 = st.date_input('Enter your birthday',
                            datetime.date(2019, 1, 1))

with optcol2:
    option6 = st.selectbox('Select your personality',
                        data2['Personality'].unique())

    option7 = st.selectbox('Select your favorite song',
                        data2['Favorite Song'].unique())

    option8 = st.selectbox('Select your favorite style',
                        data2['Style 1'].unique())

    option9 = st.selectbox('Select your favorite color',
                        data2['Color 1'].unique())

st.write("---")
st.write('You entered the following information:-')

vil1, vil2 = st.columns(2)
with vil1:
    st.write('Species: ', option1)
    st.write('Gender: ', option2)
    st.write('Hobby: ', option3)
    st.write('Catchphrase: ', option4)
    st.write('Birthday: ', option5.strftime("%B %-d"))
    st.write('Personality: ', option6)
    st.write('Favorite Song: ', option7)
    st.write('Style: ', option8)
    st.write('Color: ', option9)
with vil2:
    # filter random villagers
    random_rows = data2.sample(n=1)
    st.caption(random_rows['nh_details.quote'].values[0] + ' - ' + random_rows['Name'].values[0])

def getSign(birthDate):
    #get day of year of birth
    birthDate = dt.combine(birthDate, dt.min.time())
    dayOfBirth = (birthDate - dt(birthDate.year, 1, 1)).days

    #adjust for leap years
    if birthDate.year%4==0 and dayOfBirth > 60:
        dayOfBirth -= 1

    #build dict of max day for each sign. capricorn  is set twice due to straddling of solar year
    signs = {20:'Capricorn', 49:'Aquarius', 79:'Pisces', 109:'Aries', 140:'Taurus', 171:'Gemini', 
             203:'Cancer', 234:'Leo', 265:'Virgo', 295:'Libra', 325:'Scorpio', 355:'Sagitarius', 365:'Capricorn'}

    #create numpy array of maximum days
    daysArray = np.array(list(signs.keys()))

    #get sign max days closest to but larger than dayOfBirth
    maxDayCount = min(daysArray[daysArray >= dayOfBirth])

    return signs[maxDayCount]

# add new row to dataframe
df2.loc[-1] = ['NewVillager', option1, option2, option3, option6,
              option4, option7, option5.strftime("%B %-d"),
              option8, '', option9, '', 
              option5.strftime("%B"), getSign(option5), '']

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

df2['soup'] = df2[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['Name'])

def get_recommendations(name, cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    vill_indices = [i[0] for i in sim_scores]
    return df2['Name'].iloc[vill_indices]

villagers = get_recommendations('NewVillager', cosine_sim)

st.write("---")
st.write('Here are the villagers that you might like:')
st.write(data2[data2['Name'].isin(villagers)][main_out])

lst_vill = list(villagers)

tablst = st.tabs(lst_vill)

for i in range(len(tablst)):
    with tablst[i]:
        st.caption(data2[data2['Name'] == lst_vill[i]]['nh_details.quote'].values[0])
        col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
        with col1:
            st.image(data2[data2['Name'] == lst_vill[i]]['nh_details.image_url'].values[0], width=150)
        with col2:
            st.image(data2[data2['Name'] == lst_vill[i]]['nh_details.photo_url'].values[0], width=150)
        with col3:
            st.image(data2[data2['Name'] == lst_vill[i]]['nh_details.icon_url'].values[0], width=150)
        with col4:
            st.image(data2[data2['Name'] == lst_vill[i]]['nh_details.house_interior_url'].values[0], width=300)
        with col5:
            st.image(data2[data2['Name'] == lst_vill[i]]['nh_details.house_exterior_url'].values[0], width=300)

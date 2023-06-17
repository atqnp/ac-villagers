import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('acnh_villagers.csv')

st.image('https://dodo.ac/np/images/5/52/NH_Logo_English.png', width=200)
st.title('Villagers Recommendation')
random_rows = data.sample(n=1)
st.caption(random_rows['nh_details.quote'].values[0] + ' - ' + random_rows['Name'].values[0])

st.write('---')
st.write('Select the filters that you wish to apply:')
filtcol1, filtcol2 = st.columns(2)
with filtcol1:
    filter1 = st.selectbox('Species filter', 
                        [None] + list(data['Species'].unique()))
with filtcol2:
    filter2 = st.selectbox('Gender filter',
        [None, "Male", "Female"])
    
# data multiple filter
if filter1 is not None:
    data = data[data['Species'] == filter1]
if filter2 is not None:
    data = data[data['Gender'] == filter2]

data['PersonalitySub'] = data['Personality'] + data['nh_details.sub-personality']
data['Birthday'] = data['birthday_month'].astype(str) + ' ' + data['birthday_day'].astype(str)

df = data.copy()

main_out = ['Name', 'Species', 'Gender', 'Hobby', 'Birthday', 
            'Personality', 'Catchphrase', 'Favorite Song',
            'Style 1', 'Style 2', 'Color 1', 'Color 2']

# st.write(data[main_out])
st.write('---')
option = st.selectbox(
    'Select a villager',
    data['Name'].unique())

st.write("---")
st.write('You selected:', option)

# st.write(data[data['Name'] == option][main_out])
st.caption(data[data['Name'] == option]['nh_details.quote'].values[0])

vil1, vil2 = st.columns(2)
with vil1:
    st.image(data[data['Name'] == option]['nh_details.image_url'].values[0], width=150)
with vil2:
    st.write('Species: ', data[data['Name'] == option]['Species'].values[0])
    st.write('Gender: ', data[data['Name'] == option]['Gender'].values[0])
    st.write('Hobby: ', data[data['Name'] == option]['Hobby'].values[0])
    st.write('Birthday: ', data[data['Name'] == option]['Birthday'].values[0])
    st.write('Personality: ', data[data['Name'] == option]['Personality'].values[0])
    st.write('Catchphrase: ', data[data['Name'] == option]['Catchphrase'].values[0])
    st.write('Favorite Song: ', data[data['Name'] == option]['Favorite Song'].values[0])
    st.write('Style: ', data[data['Name'] == option]['Style 1'].values[0] + ', ' + data[data['Name'] == option]['Style 2'].values[0])
    st.write('Color: ', data[data['Name'] == option]['Color 1'].values[0] + ', ' + data[data['Name'] == option]['Color 2'].values[0])


features = ['Species', 'Gender', 'Hobby', 'PersonalitySub', 'Catchphrase',
            'Favorite Song', 'Style 1', 'Style 2', 'Color 1', 'Color 2', 
            'birthday_month', 'sign', 'Birthday', 'nh_details.quote']

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
    df[feature] = df[feature].apply(clean_data)

df['soup'] = df[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
df = df.reset_index()
indices = pd.Series(df.index, index=df['Name'])

def get_recommendations(name, cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    vill_indices = [i[0] for i in sim_scores]
    return df['Name'].iloc[vill_indices]

villagers = get_recommendations(option, cosine_sim)

st.write("---")
st.write('Here are the villagers that you might like:')
st.write(data[data['Name'].isin(villagers)][main_out])

lst_vill = list(villagers)

tablst = st.tabs(lst_vill)

for i in range(len(tablst)):
    with tablst[i]:
        st.caption(data[data['Name'] == lst_vill[i]]['nh_details.quote'].values[0])
        col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
        with col1:
            st.image(data[data['Name'] == lst_vill[i]]['nh_details.image_url'].values[0], width=150)
        with col2:
            st.image(data[data['Name'] == lst_vill[i]]['nh_details.photo_url'].values[0], width=150)
        with col3:
            st.image(data[data['Name'] == lst_vill[i]]['nh_details.icon_url'].values[0], width=150)
        with col4:
            st.image(data[data['Name'] == lst_vill[i]]['nh_details.house_interior_url'].values[0], width=300)
        with col5:
            st.image(data[data['Name'] == lst_vill[i]]['nh_details.house_exterior_url'].values[0], width=300)

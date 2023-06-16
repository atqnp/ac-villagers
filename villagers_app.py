import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.image('https://dodo.ac/np/images/5/52/NH_Logo_English.png', width=200)
st.title('Villagers Recommendation')

data = pd.read_csv('acnh_villagers.csv')

data['PersonalitySub'] = data['Personality'] + data['nh_details.sub-personality']
data['Birthday'] = data['birthday_month'].astype(str) + ' ' + data['birthday_day'].astype(str)

df = data.copy()

main_out = ['Name', 'Species', 'Gender', 'Hobby', 'Birthday', 
            'Personality', 'Catchphrase', 'Favorite Song',
            'Style 1', 'Style 2', 'Color 1', 'Color 2']

# st.write(data[main_out])

option = st.selectbox(
    'Select a villager',
    data['Name'].unique())

st.write('You selected:', option)

st.write(data[data['Name'] == option][main_out])
st.caption(data[data['Name'] == option]['nh_details.quote'].values[0])

st.image(data[data['Name'] == option]['nh_details.image_url'].values[0], width=150)

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

st.write('Here are the villagers that you might like:')
st.write(data[data['Name'].isin(villagers)][main_out])

lst_vill = list(villagers)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(lst_vill)

with tab1:
    st.caption(data[data['Name'] == lst_vill[0]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[0]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[0]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[0]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[0]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[0]]['nh_details.house_exterior_url'].values[0], width=200)
with tab2:
    st.caption(data[data['Name'] == lst_vill[1]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[1]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[1]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[1]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[1]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[1]]['nh_details.house_exterior_url'].values[0], width=200)
with tab3:
    st.caption(data[data['Name'] == lst_vill[2]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[2]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[2]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[2]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[2]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[2]]['nh_details.house_exterior_url'].values[0], width=200)
with tab4:
    st.caption(data[data['Name'] == lst_vill[3]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[3]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[3]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[3]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[3]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[3]]['nh_details.house_exterior_url'].values[0], width=200)
with tab5:
    st.caption(data[data['Name'] == lst_vill[4]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[4]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[4]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[4]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[4]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[4]]['nh_details.house_exterior_url'].values[0], width=200)
with tab6:
    st.caption(data[data['Name'] == lst_vill[5]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[5]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[5]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[5]]['nh_details.icon_url'].values[0], width=200)
    with col4:
        st.image(data[data['Name'] == lst_vill[5]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[5]]['nh_details.house_exterior_url'].values[0], width=200)
with tab7:
    st.caption(data[data['Name'] == lst_vill[6]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[6]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[6]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[6]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[6]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[6]]['nh_details.house_exterior_url'].values[0], width=200)
with tab8:
    st.caption(data[data['Name'] == lst_vill[7]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[7]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[7]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[7]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[7]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[7]]['nh_details.house_exterior_url'].values[0], width=200)
with tab9:
    st.caption(data[data['Name'] == lst_vill[8]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[8]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[8]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[8]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[8]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[8]]['nh_details.house_exterior_url'].values[0], width=200)
with tab10:
    st.caption(data[data['Name'] == lst_vill[9]]['nh_details.quote'].values[0])
    col1, col2, col3, col4, col5 = st.tabs(['Image', 'Photo', 'Icon', 'House Interior', 'House Exterior'])
    with col1:
        st.image(data[data['Name'] == lst_vill[9]]['nh_details.image_url'].values[0], width=150)
    with col2:
        st.image(data[data['Name'] == lst_vill[9]]['nh_details.photo_url'].values[0], width=150)
    with col3:
        st.image(data[data['Name'] == lst_vill[9]]['nh_details.icon_url'].values[0], width=150)
    with col4:
        st.image(data[data['Name'] == lst_vill[9]]['nh_details.house_interior_url'].values[0], width=200)
    with col5:
        st.image(data[data['Name'] == lst_vill[9]]['nh_details.house_exterior_url'].values[0], width=200)


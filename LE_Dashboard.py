#Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

######################### IMPORT DATA

########### Analysis data
df_le  = pd.read_csv('LE_Dash_Data.csv')
df_le = df_le.sort_values(["Year", "Country"], ascending=True)

########### Model & Prediction data
data_df_original = pd.read_csv('Life Expectancy Data.csv', sep = ",")

######################### DATA MANIPULATION FOR MODELING

#Copy the orginal dataframe
data_df = data_df_original.copy()

#delete feautures
data_clean=data_df.drop(['Year','Country','percentage expenditure','GDP','Population',' thinness  1-19 years',' thinness 5-9 years',' HIV/AIDS'], axis=1)

#Transform Year into numeric ordinal datatype
#data_clean[['Year']] =data_clean[['Year']].replace(to_replace={'2000':0,'2001':1,'2002':2,'2003':3,'2004':4,'2005':5,'2006':6,'2007':7,'2008':8,'2009':9,'2010':10,'2011':11,'2012':12,'2013':13,'2014':14,'2015':15})
#Transform Status into numeric
data_clean[['Status']] =data_clean[['Status']].replace(to_replace={'Developing':0,'Developed':1})

#Fill NaN with mean values
for featureimputationmean in [
    'Alcohol',' BMI ' ,'Income composition of resources'
  
]:
 data_clean[featureimputationmean].fillna(data_clean[featureimputationmean].mean(),inplace=True)
#,'GDP'

#Fill NaN with median values
for featureimputationmedian in [
    'Life expectancy ','Adult Mortality','Hepatitis B','Schooling','Polio' ,'Total expenditure','Diphtheria '
]:
    data_clean[featureimputationmedian].fillna(data_clean[featureimputationmedian].mean(),inplace=True)

# Convert Life expectancy the float column to an integer column
data_clean['Life expectancy '] = data_clean['Life expectancy '].astype(int)


######################### TRAIN TEST SPLIT

#This is time series data, always make sure you don't train on future data
# Import the train_test_split function, use shuffle=False and stratify=None
# Use test_size=0.25
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Load the input data
X = data_clean.drop(["Life expectancy "], axis=1)

# Load the target (What we trying to predict)
y = data_clean["Life expectancy "]


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


########################## TRAIN MODEL & EVALUATE 


from sklearn.neighbors import KNeighborsRegressor

#instantiate the model
regr_modelknn = KNeighborsRegressor(n_neighbors=20)
# .fit() the model
regr_modelknn.fit(X_train, y_train)

# Compute predictions on test data
y_test_predictions = regr_modelknn.predict(X_test)

########################## FIGURES (NO CALLBACK)

#Map/Choropleth

fig2 = px.choropleth(df_le, locations="Country Code3", animation_frame="Year",animation_group="Continent",
                    color="Life expectancy", 
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma, template='plotly_white')
fig2.update_coloraxes(cmin = 35,
                    cmax = 95)
fig2.update_layout(autosize=True, width=900, height=600)


########################## INITIALIZE APP

app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

load_figure_template('LUX')



app.layout = html.Div(
    id="app-container",
    children=[
        #Banner
        html.Div(
            id="banner",
            className="Banner",
            children=[html.Div(
                dbc.Nav(
                    html.H2('Life Expectancy Analysis', style={'fontSize': '35px','margin-left':'20px'})), 
                    style={'backgroundColor':'#000000', 'height': '150%','padding': '18px','color': 'white'})]
        ),
        html.Div(
            id="blurb",
            className="blurb",
            children= [ html.Div(
                dbc.Nav(
                    html.H3('Interactive dashboard for life expectancy analysis & predition', style={'fontSize': '20px','margin-left':'22px'}))
                    , style={'backgroundColor':'#000000', 'height': '100%','padding': '18px','color': 'white'})]
        ),
        html.Br(),
        html.Div([
            html.H3('Relationship between Life Expectancy and user selected feature'),
            html.P("Use the radio buttons below to change the y axis on the graph below. Then press the 'play' button to see the relationship play out over time"),
                dcc.RadioItems(
                    id='radio',
                    options= list(df_le.drop(['Country','Year','Life expectancy','thinness 10-19 years','thinness 5-9 years','Country Code2','Country Code3'], axis=1).columns.values),
                    value= list(df_le.drop(['Country','Year','Life expectancy','thinness 10-19 years','thinness 5-9 years','Country Code2','Country Code3'], axis=1).columns.values)[16],
                    inline=True,
                    style={'width':'100%', "padding":"10px"}
                ),
                dcc.Graph(id="bubble-graph")
        ], style={'margin-left':'50px','margin-right':'50px'}),
        html.Br(),
        html.Br(),
        dbc.Row(children = [
            dbc.Col(
                html.Div([
                    html.H4('Life expentancy - Map View'),
                    dcc.Graph(
                        figure=fig2, style={'display': 'inline-block'})
                ],style={"padding":"10px"}) 
        ),
            dbc.Col(
                html.Div([
                    html.H3('Life expentancy trend of countries per continents'),
                    dcc.Graph(id="graph", style={'display': 'inline-block'}),
                    dcc.Checklist(
                        id="checklist",
                        options=df_le['Continent'].unique(),
                        value=df_le['Continent'].unique()[4:5],
                        inline=True,
                        style={'width':'100%',  "padding":"10px"}) 
                ])

            )                   
        ], style={'margin-left':'50px','margin-right':'50px'}),
        html.Br(),
        html.Br(),
        dbc.Nav([
            html.H3('Predict Life expectancy for a country using key predictors',style = {"text-align":"center", 'fontSize': '25px', 'margin-left':'20px'}),
            html.P('Use the inputs below to predict the impact on life expectancy given a number of key features. The two columns below allow you to compare the predictions of life expectancy for two countries. ',style = {"text-align":"center", 'margin-left':'22px'}), 
            html.P("**** Some of the features have fairly little influence on the prediction. You will see this when you move the slider but see little to no change in life expectancy ." ,style = {"text-align":"center","font-weight": "italic", 'margin-left':'22px'})], 
            style={"text-align":"center",'backgroundColor':'#63676B', 'height': '150%','padding': '18px','color': 'white'}),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3('Country 1',style = {"text-align":"center"}),
                    #Status
                    html.H6('Status'),
                    html.P('Select country status'),
                    dcc.RadioItems(
                    id='status-radio1',
                    options= [{"label":html.Div('Developing'),"value":data_clean['Status'].unique()[0]},{"label":html.Div('Developed'),"value":data_clean['Status'].unique()[1]}],
                    value= data_clean['Status'].unique()[1],
                    inline=True,
                    style={'width':'100%', "padding":"10px"}
                    ),
                    # Adult Mortality rate
                    html.H6('Adult Mortality rate'),
                    html.P('Select number of people | per 1000 population | Between Ages 15-60 '),
                    dcc.Slider(0,800, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='admort-slider1',
                    value = 200
                    ),
                    #Infant deaths
                    html.H6('Infant deaths'),
                    html.P('Select number of people | per 1000 population '),
                    dcc.Slider(0,1000, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='infdeath-slider1',
                    value = 100
                    ),
                    #Alcohol Consumption
                    html.H6('Alcohol Consumption'),
                    html.P('Select alcohol consumption value | litres of pure alcohol '),
                    dcc.Slider(0,20, step=1,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='alc-slider1',
                    value = 4
                    ),
                    #Hepatitis B
                    html.H6('Hepatitis B'),
                    html.P('Select Hepatitis B immunization coverage percentage(%) | Among 1-year-olds '),
                    dcc.Slider(0,100, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='hep-slider1',
                    value = 80
                    ),
                    #Measles
                    html.H6('Measles'),
                    html.P('Select Measles cases value | per 100 000 people '),
                    dcc.Slider(0,1000, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='meas-slider1',
                    value = 40
                    ),
                    #BMI
                    html.H6('BMI'),
                    html.P('Select average BMI | total population '),
                    dcc.Slider(0,90, step=1,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='bmi-slider1',
                    value = 30
                    ),
                    #Under 5 deaths
                    html.H6('Under 5 deaths'),
                    html.P('Select number of under 5 deaths | per 10 000 '),
                    dcc.Slider(0,10000, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='5death-slider1',
                    value = 300
                    ),
                    #Polio
                    html.H6('Polio'),
                    html.P('Select Polio immunization coverage percentage(%) | Among 1-year-olds '),
                    dcc.Slider(0,100, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='pol-slider1',
                    value = 65
                    ),
                    #Total Expnditure
                    html.H6('Total Expnditure'),
                    html.P('Select General government expenditure on health | percentage(%) of total government expenditure '),
                    dcc.Slider(0,100, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='exp-slider1',
                    value = 60
                    ),
                    #Diphtheria
                    html.H6('Diphtheria'),
                    html.P('Select Diphtheria immunization coverage percentage(%) | Among 1-year-olds '),
                    dcc.Slider(0,100, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='dip-slider1',
                    value = 60
                    ),
                    #Income composition of resources 
                    html.H6('Income composition of resources '),
                    html.P('Select Human Development Index in terms of income composition of resources | index range from 0 to 1 '),
                    dcc.Slider(0,1, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='inc-slider1',
                    value = 0.4
                    ),
                    #Schooling 
                    html.H6('Schooling '),
                    html.P('Select number of years of schooling | years '),
                    dcc.Slider(0,21, step=1,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='sch-slider1',
                    value = 14
                    ),
                    html.Br(),
                    # Prediction
                    dbc.Card([dbc.CardHeader(html.H4("Country 1", style={"text-align":"center"})),dbc.CardBody([html.H5("Predicted Life Expectancy (years)", style={"text-align":"center", 'position':'sticky','top':0}, className="card-title",id="prediction1")])])
                ], style={'width':'100%', "padding":"10px", 'backgroundColor':'#F0F4F7', 'position':'sticky','top':0}) 
            ]),
            dbc.Col([
                html.Div([
                    html.H3('Country 2',style = {"text-align":"center"}),
                    #Status
                    html.H6('Status'),
                    html.P('Select country status'),
                    dcc.RadioItems(
                    id='status-radio2',
                    options= [{"label":html.Div('Developing'),"value":data_clean['Status'].unique()[0]},{"label":html.Div('Developed'),"value":data_clean['Status'].unique()[1]}],
                    value= data_clean['Status'].unique()[0],
                    inline=True,
                    style={'width':'100%', "padding":"10px"}
                    ),
                    # Adult Mortality rate
                    html.H6('Adult Mortality rate'),
                    html.P('Select number of people | per 1000 population | Between Ages 15-60 '),
                    dcc.Slider(0,800, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='admort-slider2',
                    value = 400
                    ),
                    #Infant deaths
                    html.H6('Infant deaths'),
                    html.P('Select number of people | per 1000 population '),
                    dcc.Slider(0,1000, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='infdeath-slider2',
                    value = 500
                    ),
                    #Alcohol Consumption
                    html.H6('Alcohol Consumption'),
                    html.P('Select alcohol consumption value | litres of pure alcohol '),
                    dcc.Slider(0,20, step=1,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='alc-slider2',
                    value = 10
                    ),
                    #Hepatitis B
                    html.H6('Hepatitis B'),
                    html.P('Select Hepatitis B immunization coverage percentage(%) | Among 1-year-olds '),
                    dcc.Slider(0,100, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='hep-slider2',
                    value = 50
                    ),
                    #Measles
                    html.H6('Measles'),
                    html.P('Select Measles cases value | per 100 000 people '),
                    dcc.Slider(0,1000, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='meas-slider2',
                    value = 400
                    ),
                    #BMI
                    html.H6('BMI'),
                    html.P('Select average BMI | total population '),
                    dcc.Slider(0,90, step=1,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='bmi-slider2',
                    value = 40
                    ),
                    #Under 5 deaths
                    html.H6('Under 5 deaths'),
                    html.P('Select number of under 5 deaths | per 10 000 '),
                    dcc.Slider(0,10000, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='5death-slider2',
                    value = 900
                    ),
                    #Polio
                    html.H6('Polio'),
                    html.P('Select Polio immunization coverage percentage(%) | Among 1-year-olds '),
                    dcc.Slider(0,100, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='pol-slider2',
                    value = 40
                    ),
                    #Total Expnditure
                    html.H6('Total Expnditure'),
                    html.P('Select General government expenditure on health | percentage(%) of total government expenditure '),
                    dcc.Slider(0,100, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='exp-slider2',
                    value = 40
                    ),
                    #Diphtheria
                    html.H6('Diphtheria'),
                    html.P('Select Diphtheria immunization coverage percentage(%) | Among 1-year-olds '),
                    dcc.Slider(0,100, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='dip-slider2',
                    value = 60
                    ),
                    #Income composition of resources 
                    html.H6('Income composition of resources '),
                    html.P('Select Human Development Index in terms of income composition of resources | index range from 0 to 1 '),
                    dcc.Slider(0,1, step=None,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='inc-slider2',
                    value = 0.6
                    ),
                    #Schooling 
                    html.H6('Schooling '),
                    html.P('Select number of years of schooling | years '),
                    dcc.Slider(0,21, step=1,  marks=None, tooltip={"placement": "bottom", "always_visible": True},
                    id='sch-slider2',
                    value = 12
                    ),
                    html.Br(),
                    # Prediction
                    dbc.Card([dbc.CardHeader(html.H4("Country 2", style={"text-align":"center"})),dbc.CardBody([html.H5("Predicted Life Expectancy (years)", style={"text-align":"center", 'position':'sticky','top':0}, className="card-title",id="prediction2")])])
                ], style={'width':'100%', "padding":"10px", 'backgroundColor':'#D2E9F9'}) 
            ])                
    ], style = {'margin-left':'50px','margin-right':'50px'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(
                dbc.Nav(
                    html.P('Dashboard & Model developed by Nadia Ebrahim (27723003) & Elelwani Mafela (27684393)', style={'fontSize': '20px',"text-align":"centre"})) 
                    , style={'backgroundColor':'#2F2F2F', 'height': '100%','padding': '18px','color': 'white'})
        
],style = {'margin-left':'0px', 'margin-top':'7px'})


########################## CALLBACKS

# Bubble map - radio buttons

@app.callback(
    Output('bubble-graph', 'figure'), 
    [Input('radio', 'value')])
def toggle_x_axis(features):
    fig1 = px.scatter(df_le,
                    x="Life expectancy", y=features,  color="Continent", size="Population", animation_frame="Year",animation_group="Country", hover_name="Country",
                    log_x=False, size_max=100,
                    template='plotly_white')
    return fig1

# Line graph
@app.callback(
    Output('graph', 'figure'), 
    [Input('checklist', 'value')])
def update_line_chart(continents):
    df_conti = df_le[df_le.Continent.isin(continents)]
    fig3 = px.line(df_conti, 
        x="Year", y="Life expectancy", color='Country',template='plotly_white')

    fig3.update_layout(autosize=True, width=900, height=600)
    return fig3

# Country 1 prediction

@app.callback(
    Output('prediction1', 'children'),
    [Input('status-radio1', 'value'),
    Input('admort-slider1', 'value'),
    Input('infdeath-slider1', 'value'),
    Input('alc-slider1', 'value'),
    Input('hep-slider1', 'value'),
    Input('meas-slider1', 'value'),
    Input('bmi-slider1', 'value'),
    Input('5death-slider1', 'value'),
    Input('pol-slider1', 'value'),
    Input('exp-slider1', 'value'),
    Input('dip-slider1', 'value'),
    Input('inc-slider1', 'value'),
    Input('sch-slider1', 'value')])
def first_prediction(status,admortality,infdeaths,alcohol,hepatitis,measles,bmi,und5deaths,polio,expend,diphth,income,school):
    # Assign Inputs and create dataframe
    
    modelled_columns = {'Status': pd.Series(dtype='int'), 
        'Adult Mortality': pd.Series(dtype='float'), 
        'infant deaths': pd.Series(dtype='int'), 
        'Alcohol': pd.Series(dtype='float'), 
        'Hepatitis B': pd.Series(dtype='float'), 
        'Measles ': pd.Series(dtype='int'), 
        ' BMI ': pd.Series(dtype='float'), 
        'under-five deaths ': pd.Series(dtype='int'), 
        'Polio': pd.Series(dtype='float'), 
        'Total expenditure': pd.Series(dtype='float'), 
        'Diphtheria ': pd.Series(dtype='float'), 
        'Income composition of resources': pd.Series(dtype='float'), 
        'Schooling': pd.Series(dtype='float')}

    input_data = [[status,admortality,infdeaths,alcohol,hepatitis,measles,bmi,und5deaths,polio,expend,diphth,income,school]]

    pred1_df = pd.DataFrame(data=input_data, columns = modelled_columns, index=[0])

    # Predict
    pred1 = regr_modelknn.predict(pred1_df).astype('str')

    return pred1


# Country 2 prediction

@app.callback(
    Output('prediction2', 'children'),
    [Input('status-radio2', 'value'),
    Input('admort-slider2', 'value'),
    Input('infdeath-slider2', 'value'),
    Input('alc-slider2', 'value'),
    Input('hep-slider2', 'value'),
    Input('meas-slider2', 'value'),
    Input('bmi-slider2', 'value'),
    Input('5death-slider2', 'value'),
    Input('pol-slider2', 'value'),
    Input('exp-slider2', 'value'),
    Input('dip-slider2', 'value'),
    Input('inc-slider2', 'value'),
    Input('sch-slider2', 'value')])
def first_prediction(status,admortality,infdeaths,alcohol,hepatitis,measles,bmi,und5deaths,polio,expend,diphth,income,school):
    # Assign Inputs and create dataframe
    modelled_columns = {'Status': pd.Series(dtype='int'), 
        'Adult Mortality': pd.Series(dtype='float'), 
        'infant deaths': pd.Series(dtype='int'), 
        'Alcohol': pd.Series(dtype='float'), 
        'Hepatitis B': pd.Series(dtype='float'), 
        'Measles ': pd.Series(dtype='int'), 
        ' BMI ': pd.Series(dtype='float'), 
        'under-five deaths ': pd.Series(dtype='int'), 
        'Polio': pd.Series(dtype='float'), 
        'Total expenditure': pd.Series(dtype='float'), 
        'Diphtheria ': pd.Series(dtype='float'), 
        'Income composition of resources': pd.Series(dtype='float'), 
        'Schooling': pd.Series(dtype='float')}

    input_data = [[status,admortality,infdeaths,alcohol,hepatitis,measles,bmi,und5deaths,polio,expend,diphth,income,school]]

    pred2_df = pd.DataFrame(data=input_data, columns = modelled_columns, index=[0])
    

    # Predict
    pred2 = regr_modelknn.predict(pred2_df).astype('str')

    return pred2


#Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
# --- Import Required Libraries ---
# Dash for web application framework, data visualization, and interactive components
# Pandas and NumPy for data manipulation and numerical operations
# Requests for API calls, Plotly for visualization, and additional libraries for modeling and forecasting
import dash
import numpy as np
import requests
from dash import dcc, html, callback, Dash 
from dash.dependencies import Output, Input
import pandas as pd
import dash_bootstrap_components as dbc
import datetime
import plotly.express as px
import plotly.graph_objects as go
import os 
import matplotlib.pylab as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from joblib import dump, load
import xml.etree.ElementTree as ET

# --- Load Price Data ---
# Load historical electricity price data for analysis and modeling
# Ensure the index is set to datetime for time series operations
path = r"C:\Users\ddoko\OneDrive\Documents 1\Career\Resources\Trading\Power_Trading_Dashboard\price_df.csv"
price_df = pd.read_csv(path, index_col=0)
price_df.index = pd.to_datetime(price_df.index)

# --- Helper Functions ---
def add_lags(df, price_df):
    """
    Add lag features to capture past price trends for modeling purposes.
    
    Parameters:
        df (DataFrame): DataFrame to which lag features are added.
        price_df (DataFrame): DataFrame containing historical price data.
    
    Returns:
        DataFrame: Updated DataFrame with lag features.
    """
    target_map = price_df['Price (EUR/MWhe)'].to_dict()
    # Map past prices at 1-year, 2-year, and 3-year intervals as lag features
    lag1 = (df.index - pd.Timedelta('364 days')).map(target_map)
    lag2 = (df.index - pd.Timedelta('728 days')).map(target_map)
    lag3 = (df.index - pd.Timedelta('1092 days')).map(target_map)
    df.insert(0, "lag1", lag1, True)
    df.insert(1, "lag2", lag2, True)
    df.insert(2, "lag3", lag3, True)    
    return df

def get_country_temp(country, delta):
    """
    Fetch weather data for a specified country and process it into a structured DataFrame with time-based features and lags.
    
    Parameters:
    -----------
    country : str
        The name of the country for which weather data is being fetched. Currently supports 'France' only.
        
    delta : int
        The number of days for which weather data is needed:
        - delta=0 fetches weather data for the remaining hours of the current day.
        - delta>0 fetches data for the specified number of days.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing weather data with time features and lags, ready for time-series analysis.
        
    """

    # Changes latitude and longitude based on the specified country
    if country == 'France':
        lat = 48.857548
        lon = 2.351377  

    # Determine how many rows to retrieve from the API call
    if delta == 0:
        count = 24 - (datetime.datetime.now().hour)
    else:
        count = 24 * delta

    # Make API call to retrieve weather data
    API_key = r"08aaef8862165384f3748755a2a0dd9b"
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_key}&cnt={count}"
    response = requests.get(url)
    if response.status_code == 200:
        current_temp = response.json()  # Parse the JSON response
    else:
        print(f"Error: {response.status_code}")

    # Calculate start and end times for data indexing
    global current_time
    current_time = pd.Timestamp.now()  # Capture current timestamp
    start_time = "2015-01-01 00:00:00"
    if delta == 0:
        end_time = pd.Timestamp(current_time.strftime('%Y-%m-%d') + ' 23:00:00')
    else:
        end_time = current_time + pd.Timedelta(f'{delta} days')
    
    # Create a DataFrame with a complete hourly time index
    day_index = pd.date_range(start=start_time, end=end_time, freq='h')
    
    # Define column names for weather data and initialize DataFrame
    columns_list = ['temp', 'rain_3h', 'wind_speed', 'visibility', 'pressure', 'humidity', 'feels_like']
    base_current_df = pd.DataFrame(columns=columns_list, index=day_index)
    current_df = base_current_df.copy()

    # Add time-related features to the DataFrame
    current_df['Hour'] = current_df.index.hour
    current_df['Day'] = current_df.index.day
    current_df['Month'] = current_df.index.month
    current_df['Year'] = current_df.index.year
    current_df['Day_of_Week'] = current_df.index.weekday
    current_df['Day_of_Year'] = current_df.index.dayofyear
    
    # Add seasonal components using Fourier series and other features
    fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for annual seasonality

    dp = DeterministicProcess(
        index=current_df.index,
        constant=True,               # Dummy feature for bias (y-intercept)
        order=1,                     # Trend (order 1 means linear)
        seasonal=True,               # Weekly seasonality (indicators)
        additional_terms=[fourier],  # Annual seasonality (Fourier terms)
        drop=True,                   # Drop terms to avoid collinearity
    )

    # Generate seasonal features and merge them with current data
    Season = dp.in_sample()
    current_df = pd.concat([Season, current_df], axis=1)

    # Filter DataFrame to retain rows within the specified timeframe
    current_df = current_df.loc[(current_df.index >= current_time) & (current_df.index <= end_time)]
    current_df = add_lags(current_df, price_df)  # Add lag features based on external data

    # Map API response data to the DataFrame
    for x in range(0, len(current_temp['list'])):
        time_txt = current_temp['list'][x]['dt_txt']
        timestampp = pd.to_datetime(time_txt).strftime('%Y-%m-%d %H:%M')
        rain = current_temp['list'][x].get('rain', {}).get('3h', 0)  # Default to 0 if 'rain' or '3h' is missing
        visibility = current_temp['list'][x].get('visibility', 0)  # Default to 0 if visibility is missing

        current_df.loc[timestampp, columns_list] = [
            current_temp['list'][x]['main']['temp'],
            rain, 
            current_temp['list'][x]['wind']['speed'],
            visibility,
            current_temp['list'][x]['main']['pressure'],
            current_temp['list'][x]['main']['humidity'],
            current_temp['list'][x]['main']['feels_like']
        ]
    
    # Ensure numeric data types and interpolate missing values
    numeric_columns = ['temp', 'rain_3h', 'wind_speed', 'visibility', 'pressure', 'humidity', 'feels_like']
    for col in numeric_columns:
        current_df[col] = pd.to_numeric(current_df[col], errors='coerce')
    current_df = current_df.interpolate(method='time', limit_direction='both', axis=0)

    return current_df

def get_price_curve(country, delta):
    if country == "France":
        out_domain = in_domain = r"10YFR-RTE------C"
    start = datetime.datetime.now().strftime('%Y%m%d%H%M')
    if delta==0:
        end_time = pd.Timestamp(current_time.strftime('%Y-%m-%d') + ' 23:59:00')
    else:
        end_time = pd.Timestamp.now()+ pd.Timedelta(f'{delta} days')
    end = end_time.strftime('%Y%m%d%H%M')
    security_token = r"12b06457-bd0a-415f-b433-865595cf1ee3"

    url = f"https://web-api.tp.entsoe.eu/api?documentType=A44&periodStart={start}&periodEnd={end}&out_Domain={out_domain}&in_Domain={in_domain}&securityToken={security_token}"

    response = requests.get(url)

    if response.status_code == 200:
        root = ET.fromstring(response.text)
        
        # Define namespace
        namespace = {'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3'}
        
        # Initialize an empty list to store data
        data = []
        
        # Iterate through TimeSeries elements
        for timeseries in root.findall('ns:TimeSeries', namespace):
            period = timeseries.find('ns:Period', namespace)
            time_interval = period.find('ns:timeInterval', namespace)
            start_time = time_interval.find('ns:start', namespace).text
            
            # Iterate through Point elements
            for point in period.findall('ns:Point', namespace):
                position = int(point.find('ns:position', namespace).text)
                price = float(point.find('ns:price.amount', namespace).text)
                timestamp = pd.Timestamp(start_time) + pd.Timedelta(hours=position - 1)
                data.append({'timestamp': timestamp, 'price': price})
        
        # Create a DataFrame
        power_curve_df = pd.DataFrame(data)
        power_curve_df = power_curve_df.set_index('timestamp')
        power_curve_df.index = pd.to_datetime(power_curve_df.index.strftime("%Y-%m-%d %H:%M:%S"))
        
    else:
        print(f"Error: {response.status_code}")    
    price_curve = power_curve_df.loc[(power_curve_df.index>=pd.Timestamp.now())&(power_curve_df.index<=end_time)]

    return price_curve

def run_model(current_df, price_curve):
    model = load('model4.joblib')
    X_forecast = current_df
    Y_pred = pd.Series(model.predict(X_forecast), index=X_forecast.index)
    price_curve['model_price'] = Y_pred
    price_curve['diff'] = price_curve['model_price'] - price_curve['price']

    # Find the timestamps of the max and min diff
    max_diff_time = price_curve['diff'].idxmax()
    min_diff_time = price_curve['diff'].idxmin()

    # Create the figure with line plots
    fig = px.line(price_curve, x=price_curve.index, y=['price', 'model_price'],
                  labels={'value': 'Price (EUR/MWhe)', 'variable': 'Legend'},
                  title='Market Prices vs. Forecast')

    # Add circles where the max and min diff occur
    fig.add_trace(go.Scatter(
        x=[max_diff_time],
        y=[price_curve.loc[max_diff_time, 'price']],
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name='Max Diff Circle',
        hovertext='Buy here',  # Hover text for the max diff spot
        hoverinfo='text',  # Only show hover text
        showlegend=False  # Hide the circle from the legend
    ))

    fig.add_trace(go.Scatter(
        x=[min_diff_time],
        y=[price_curve.loc[min_diff_time, 'price']],
        mode='markers',
        marker=dict(size=10, color='green', symbol='circle'),
        name='Min Diff Circle',
        hovertext='Sell here',  # Hover text for the min diff spot
        hoverinfo='text',  # Only show hover text
        showlegend=False  # Hide the circle from the legend
    ))

    # Update layout with styles
    fig.update_layout(
        template='plotly_white',  # White background for a clean look
        title_x=0.5,
        xaxis_title='Time',
        yaxis_title='Price (EUR/MWhe)',
        legend_title='Data Source',
        
        # Customize axis colors and grid
        xaxis=dict(
            showgrid=True,  # Display grid lines
            gridwidth=1,  # Lighter grid lines
            gridcolor='rgba(0,0,0,0.1)',  # Light grid color
            tickangle=45,  # Angled x-axis ticks for better readability
        ),
        yaxis=dict(
            showgrid=True,  # Display grid lines
            gridwidth=1,  # Lighter grid lines
            gridcolor='rgba(0,0,0,0.1)',  # Light grid color
        ),
        
        # Styling the figure
        plot_bgcolor='white',  # White background for the plot area
        paper_bgcolor='white',  # White background for the entire figure
        font=dict(
            family='Arial, sans-serif',  # Neutral font for readability
            color='rgba(0,0,0,0.8)'  # Darker text color for contrast
        ),
        
        # Line styling for better visibility
        legend=dict(
            orientation="h",  # Horizontal legend to save space
            x=0.5,  # Center the legend
            xanchor="center",  # Center the legend horizontally
            y=-0.2,  # Position the legend below the chart
            yanchor="top",
            font=dict(
                size=12,  # Smaller legend text
                color='rgba(0, 0, 0, 0.8)'  # Darker legend text for readability
            )
        ),
        
        # Rounded corners for a sleek look
        margin=dict(t=50, b=50, l=50, r=50),  # Margins for better spacing
        height=500,  # Set a specific height for the chart
    )

    # Set the line styles
    fig.update_traces(
        line=dict(width=2),  # Thicker lines for better visibility
        hoverinfo='x+y',  # Display only x and y values on hover
    )
    
    return fig, price_curve

    
def make_strategy(price_curve):
    profit =  price_curve['diff'][price_curve['diff'].idxmax()] - price_curve['diff'][price_curve['diff'].idxmin()] 
    buy_time = price_curve['diff'].idxmax()
    sell_time = price_curve['diff'].idxmin()

    return profit, buy_time, sell_time

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    # Header
    html.Div(className='app-header', children=[
        html.H1("Power Trading Dashboard", className='display-3')
    ]),

    # Country Dropdown
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                ['France'], ['France'], multi=True,
                id="country_dropdown",
                placeholder="Select Country",
                className="dropdown-select"
            )
        ], width={'size': 6, 'offset': 3}, className='dropdown-container')
    ], className="mt-4"),  # Add top margin here

    # Chart and Strategy Side-by-Side
    dbc.Row([
        dbc.Col(dcc.Graph(id='power-price-chart', className="chart-container"), width=8),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Trading Strategy", className="card-title"),
                    dcc.Markdown("This section will have the trade strategy", id='trade-strategy', className="strategy-text"),
                    # html.P("This section will have the trade strategy", id='trade-strategy', className="strategy-text"),
                ]),
                className="strategy-card"
            )
        ], width=4, className="strategy-col")
    ], className="mt-4"),  # Add top margin here

    # Timeframe Selection Buttons
    dbc.Row([
        dbc.Col(html.Div([
            dbc.ButtonGroup([
                dbc.Button('EOD', id='btn-nclicks-1', className='btn-eod'),
                dbc.Button('1 Day', id='btn-nclicks-2', className='btn-1day'),
                dbc.Button('3 Days', id='btn-nclicks-3', className='btn-3days'),
                dbc.Button('Max', id='btn-nclicks-4', className='btn-max'),
            ], className="dash-button-group"),
            dcc.Store(id='user_time_delta')
        ]), width={'size': 10, 'offset': 1})
    ], className="mt-4")  # Add top margin here
])



# Adds functionality to display prices and strategy for selected country(s) and timeframe
# Use callback to call functions to 1.Call APIs 2.Run model 3.Make chart/trade 4.save then read from???

#Takes button clicked and returns a timedelta
@callback(
    Output('user_time_delta', 'data'),
    Input('btn-nclicks-1', 'n_clicks'),
    Input('btn-nclicks-2', 'n_clicks'),
    Input('btn-nclicks-3', 'n_clicks'),
    Input('btn-nclicks-4', 'n_clicks'))
def displayClick(btn1, btn2, btn3, btn4):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 0
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    delta = 0
    if "btn-nclicks-1" == ctx.triggered_id:
        delta = 0
    elif "btn-nclicks-2" == ctx.triggered_id:
        delta = 1
    elif "btn-nclicks-3" == ctx.triggered_id:
        delta = 3
    elif "btn-nclicks-4" == ctx.triggered_id:
        delta = 5
    return delta

#takes in the result from the country dropdown and timedelta->return a graph and strategy
@callback(
    Output('power-price-chart', 'figure'),
    Output('trade-strategy', 'children'),
    Input('user_time_delta', 'data'),
    Input('country_dropdown', 'value'))
def update_graph(user_time_delta, country_dropdown):
    if not country_dropdown:
        return {}, "Please select a country."
    
    current_df = get_country_temp(country=country_dropdown[0], delta=user_time_delta)
    price_curve = get_price_curve(country=country_dropdown[0], delta=user_time_delta)
    
    # if current_df.empty or price_curve.empty:
    #     return {}, "No data available for the selected timeframe."
    
    fig, updated_price_curve = run_model(current_df, price_curve)
    
    profit, buy_time, sell_time = make_strategy(updated_price_curve)
    strategy_text = f"""
    Profit: {profit:.2f}
    Buy Time: {buy_time}
    Sell Time: {sell_time}
    """
    
    return fig, strategy_text


if __name__ == "__main__":
    app.run_server(debug = True)
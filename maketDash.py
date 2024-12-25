import os
import numpy as np
import pandas as pd
import math

import random
import datetime
from dateutil.relativedelta import relativedelta
from babel.dates import format_date

import nltk
nltk.download('brown')
from nltk.corpus import brown

import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

from IPython.display import display

from PIL import ImageColor

from io import BytesIO
from wordcloud import WordCloud
import base64


import string
from collections import Counter

def load_stopwords_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return stopwords

file_path = 'stopwords-ru.txt'  # Укажите путь к вашему файлу со стоп-словами
puncts = set(list(string.punctuation) + ['—', '»', '«', '``', '–', "''"])
stopwords_ru = set(load_stopwords_from_file(file_path))
predlogi = set(['без' , 'в' , 'до' , 'для' , 'за' , 'из' , 'к' , 'на' , 'над' , 'о' , 'об' , 'от' , 'по' , 'под' , 'пред' , 'при' , 'про' , 'с' , 'у' , 'через']) 
souzy = set(['а' , 'и' , 'чтобы' , 'если', 'потому что' , 'как будто' , 'то есть'])
exclude = set(['например', 'какие', 'кто-то', 'что-то', 'кстати', 'многие', 'таких', 'может', 'любой', 'поэтому', 'https'])
numbers = set('1234567890')
dell_words = stopwords_ru | predlogi | souzy | numbers | exclude


# Указываем путь к папке с файлами
folder_path = os.getcwd()

# Получаем список файлов в папке
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

channels = pd.read_csv(folder_path + file_list[0])
posts = pd.read_csv(folder_path + file_list[1])
reactions = pd.read_csv(folder_path + file_list[2])
subscribers = pd.read_csv(folder_path + file_list[3])
views = pd.read_csv(folder_path + file_list[4])


def date_ago(tp, num=0):
    if tp == 'today':
        return datetime.datetime.now().strftime("%Y-%m-%d") 
    elif tp == 'yesterday':
        return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    elif tp == 'days':
        return (datetime.datetime.now() - datetime.timedelta(days=num+1)).strftime("%Y-%m-%d")
    elif tp == 'weeks':
        return (datetime.datetime.now() - datetime.timedelta(days= 7*num + 1)).strftime("%Y-%m-%d") 
    elif tp == 'months':
        return (datetime.datetime.now() - relativedelta(months=num) - datetime.timedelta(days=1)).strftime("%Y-%m-%d") 
    else:
        print('Неправильно задан тип даты или не указано количество повторений (возможные типы дат: today, yesterday, days, weeks, months')

def convert_date(date):
    try:
        return datetime.datetime.strptime(date,'%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        # Если строка не может быть преобразована в дату, возвращаем NaT (Not a Time)
        return pd.NaT

# Функция для определения градиентной заливки
def get_gradient_color(value, min_val=0, max_val=100):
    # Если значение равно нулю, возвращаем прозрачный цвет
    if value == 0:
        return "transparent"
    
    # Рассчитываем процентное соотношение между минимальным и максимальным значением
    ratio = (value - min_val) / (max_val - min_val)
    # Ограничиваем диапазон значений
    ratio = max(min(ratio, 1), 0)

     # Начальные и конечные значения RGB
    start_r, start_g, start_b = 139, 0, 0 #245, 223, 191  # Бежевый (#f5dfbf)
    end_r, end_g, end_b = 34, 139, 34          # Зелёный (#228B22)
    
    # Рассчитываем промежуточные значения RGB
    r = int(start_r * (1 - ratio) + end_r * ratio)
    g = int(start_g * (1 - ratio) + end_g * ratio)
    b = int(start_b * (1 - ratio) + end_b * ratio)
    
    color = '#%02x%02x%02x' % (r, g, b)
    return color

def create_table(post_view, max_days, channel):
    
    filtered_post_view = post_view[(post_view['days_diff'] <= max_days)&(post_view.channel_name==channel)].copy()
    filtered_post_view = filtered_post_view.groupby(['post_datetime', 'post_id'
                                                     , 'current_views', 'days_diff'])[['view_change', 'percent_new_views']].sum().reset_index()
    grouped_df = filtered_post_view.groupby(['post_datetime', 'post_id']).agg({
        'view_change': lambda x: list(x),
        'percent_new_views': lambda x: list(x),
        'current_views': lambda x: x.iloc[-1]
    }).reset_index()

    max_days = int(round(max_days))
    
    columns = ["ID поста", "Дата публикации", "Текущие просмотры"] + [f"{i} д" for i in range(1, max_days+1)]
    data = []
    
    for _, row in reversed(list(grouped_df.iterrows())):
        view_change = row['view_change']
        percent_new_views = row['percent_new_views']
        current_views = row['current_views']
        
        row_data = [
            html.Td(f"{row['post_id']}"),
            html.Td(f"{datetime.datetime.strptime(row['post_datetime'], "%Y-%m-%d %H:%M:%S.%f").strftime("%b %d, %Y")}", style={"text-align": "center"}),
            html.Td(current_views)
        ]
        for day in range(1, max_days+1):
            if day <= len(view_change):
                cell_value = f"{view_change[day-1]} ({percent_new_views[day-1]:.2f}%)"
                
                # Проверяем процентное значение
                if percent_new_views[day-1] >= 80:
                    text_color = "#228B22"  # Зеленый цвет
                else:
                    # Используем функцию для получения градиентного цвета
                    text_color = get_gradient_color(percent_new_views[day-1])
                    
                row_data.append(html.Td(cell_value, style={"color": text_color
                                                           , "font-weight": "bold"
                                                           , 'text-align': 'center'}))  # Изменение стиля текста
            else:
                row_data.append(html.Td("-", style={"text-align": "center"}))
     
        data.append(html.Tr(row_data))
        
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in columns])),
        html.Tbody(data)
    ], className="tgstat-table")



def hex_to_rgb(hex_code):
    """Преобразует HEX-код в RGB."""
    rgb = ImageColor.getcolor(hex_code, "RGB")
    return rgb

def interpolate_color(start_color, end_color, steps):
    """Интерполирует цвет между двумя значениями RGB."""
    start_r, start_g, start_b = start_color
    end_r, end_g, end_b = end_color
    step_r = (end_r - start_r) / steps
    step_g = (end_g - start_g) / steps
    step_b = (end_b - start_b) / steps
    return [(int(start_r + i * step_r),
             int(start_g + i * step_g),
             int(start_b + i * step_b)) for i in range(steps)]

def gradient_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    start_color = hex_to_rgb('#8B0000')
    end_color = hex_to_rgb('#ffb347')
    num_steps = 50  # Количество шагов равно количеству слов
    colors = interpolate_color(start_color, end_color, num_steps)
    index = random.randint(0, num_steps - 1)  # Случайное число от 0 до количества слов
    r, g, b = colors[index]
    return f"rgb({r}, {g}, {b})"



#CHANNELS

# что за переменнная channel_type varchar(50)?


#POSTS

# date - datetime or date? if datetime - change code, else - all ok

posts.rename(columns={'date': 'datetime'}, inplace=True)
posts = posts.merge(channels[['id', 'channel_name']].rename(columns={'id':'channel_id'}), on = 'channel_id', how='left')
posts.loc[:, 'date'] = pd.to_datetime(posts.datetime).apply(lambda t: t.strftime('%Y-%m-%d'))
posts.loc[:, 'time'] = posts.datetime.str[10:]
posts.loc[:, 'cnt'] = posts.groupby(['channel_id', 'date'])[['message_id']].transform('count')
posts.loc[:, 'hour'] = pd.to_datetime(posts.datetime).apply(lambda t: t.hour)
posts = posts[(~posts.text.isnull())&(posts.text != 'Нет текста')].copy()

#VIEWS
# колонка date создавалась как основа для datetime, в исходной ее не будет (проверить, можно ли далее по коду заменить все на datetime)

views.rename(columns={'timestamp': 'datetime', 'views': 'view_cnt'}, inplace=True)
views.loc[:, 'date'] = pd.to_datetime(views.datetime).apply(lambda t: t.strftime('%Y-%m-%d'))
view_change = views.sort_values(by = ['post_id', 'datetime'])\
                        .groupby('post_id')[['view_cnt']].diff()\
                        .rename(columns={'view_cnt':'view_change'})

views = views.merge(view_change, left_index = True, right_index=True)
views.loc[:, 'view_change'] = np.where(views.view_change.isnull(), views.view_cnt, views.view_change)

#SUBSCRIBERS
# колонка date создавалась как основа для datetime, в исходной ее не будет (проверить, можно ли далее по коду заменить все на datetime)

subs = subscribers.copy()
subs.rename(columns={'timestamp': 'datetime', 'subscriber_count': 'subs_cnt'}, inplace=True)

subs.loc[:, 'date'] = pd.to_datetime(subs.datetime).apply(lambda t: t.strftime('%Y-%m-%d'))

subs = subs.merge(channels[['id', 'channel_name']].rename(columns={'id':'channel_id'}), on = 'channel_id', how='left')

subs.sort_values(by=['channel_id', 'datetime'], inplace=True)
subs.loc[:, 'subs_change'] =  subs.groupby('channel_id')[['subs_cnt']].diff().fillna(0) #np.hstack((np.array([np.nan]), np.diff(subs.subs_cnt, axis=0)))

subs.loc[:, 'subs_change_pos'] = np.where(subs.subs_change>0, subs.subs_change, 0)
subs.loc[:, 'subs_change_neg'] = np.where(subs.subs_change<0, subs.subs_change, 0) 
subs.loc[:, 'day_change_pos'] = subs.groupby(['channel_id','date'])[['subs_change_pos']].transform('sum')
subs.loc[:, 'day_change_neg'] = subs.groupby(['channel_id', 'date'])[['subs_change_neg']].transform('sum')

subs.datetime = pd.to_datetime(subs.datetime)
del subscribers

#REACTIONS
# колонка date создавалась как основа для datetime, в исходной ее не будет (проверить, можно ли далее по коду заменить все на datetime)

reacts = reactions.copy()
reacts.rename(columns={'timestamp': 'datetime', 'count': 'react_cnt'}, inplace=True)

reacts.loc[:, 'date'] = pd.to_datetime(reacts.datetime).apply(lambda t: t.strftime('%Y-%m-%d'))
del reactions

#POSTS & VIEWS
post_view = views[['post_id', 'view_cnt', 'view_change','datetime']]\
                    .merge(posts[['id', 'channel_name', 'date','datetime']].rename(columns={'id':'post_id', 'datetime':'post_datetime'})
                        , on='post_id')[['channel_name', 'post_id', 'post_datetime', 'datetime', 'view_cnt', 'view_change']]\
                    .sort_values(by=['channel_name','post_id', 'datetime'])
post_view = post_view.reset_index().drop('index', axis=1)

post_view.loc[:, 'hours_diff'] = (pd.to_datetime(post_view.datetime) - pd.to_datetime(post_view.post_datetime))\
                                                                            .apply(lambda t: math.ceil(t.total_seconds()/3600))
post_view.loc[:, 'days_diff'] = (pd.to_datetime(post_view.datetime) - pd.to_datetime(post_view.post_datetime))\
                                                                            .apply(lambda t: math.ceil(t.total_seconds()/(3600*24)))

bins = list(range(0, 74))
labels = list(range(1, 74))
post_view.loc[:, 'hours_group'] = pd.cut(post_view['hours_diff'], bins=bins, labels=labels).fillna(73)

# Рассчитываем процент новых просмотров относительно общего количества просмотров
post_view['current_views'] = post_view.groupby('post_id')['view_cnt'].transform('last')
post_view['percent_new_views'] = (post_view['view_change'] / post_view['current_views']) * 100
post_view = post_view.sort_values(by=['channel_name', 'post_datetime'], ascending=False)


#POSTS & VIEWS & REACTIONS
group_reacts = reacts.groupby(['post_id', 'reaction_type'])[['datetime', 'react_cnt']].last().reset_index()
group_post_view = post_view.groupby(['channel_name', 'post_datetime','post_id',  'current_views'])[['datetime']].last().reset_index()
#date_format
group_reacts.loc[:, 'datetime_format'] = group_reacts.datetime.apply(lambda date: convert_date(date).strftime('%Y-%m-%d %H:%M:%S'))
group_post_view.loc[:, 'datetime_format'] = group_post_view.datetime.apply(lambda date: convert_date(date).strftime('%Y-%m-%d %H:%M:%S'))
#drop
group_reacts.drop('datetime', axis=1, inplace=True)
group_post_view.drop('datetime', axis=1, inplace=True)
#merge & create exclude lists
gr_pvr = group_post_view.merge(group_reacts, on = ['post_id', 'datetime_format'], how='left').drop_duplicates()
noreact = gr_pvr[gr_pvr.react_cnt.isnull()].post_id.unique()  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
gr_pvr = gr_pvr[~gr_pvr.post_id.isin(noreact)].copy()
no_post_have_react = list(set(group_reacts.post_id) - set(gr_pvr.post_id)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#group_reacts.shape[0] - gr_pvr.shape[0] #????????? where 7 
#
#new fields
gr_pvr.loc[:, 'react_cnt_sum'] = gr_pvr.groupby('post_id')[['react_cnt']].transform('sum')
gr_pvr.loc[:, 'idx_active'] = round(gr_pvr.react_cnt_sum/gr_pvr.current_views*100,2)


#-----------------------------Метрики по подписчикам-------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

def calculate_mean_max_subs(channel):
    filtered_df = subs[subs.channel_name==channel][['date', 'day_change_pos', 'day_change_neg']].drop_duplicates()
    
    # вопрос по округлению!!!!!!!
    mean_subs_pos, mean_subs_neg = int(round(filtered_df.day_change_pos.mean(), 0)), int(round(filtered_df.day_change_neg.mean(), 0)) 
    max_subs_pos, max_subs_neg = int(round(filtered_df.day_change_pos.max(), 0)), int(round(filtered_df.day_change_neg.min(), 0)) 
    
    # Средний ежедневный прирост
    # Средний ежедневный отток    
    # Максимальный дневной прирост 
    # Максимальный дневной отток
    
    return mean_subs_pos, mean_subs_neg, max_subs_pos, max_subs_neg

#-----------------------------Метрики по публикациям-------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

def calculate_mean_posts(channel):
    filtered_df = posts[posts.channel_name==channel].copy()
    filtered_df.loc[:, 'date_week'] = pd.to_datetime(filtered_df.date).apply(lambda d: d.isocalendar().week)
    filtered_df.loc[:, 'date_month'] = filtered_df.date.str[:7]

    mean_posts_day = int(round(filtered_df.cnt.sum()/len(pd.date_range(filtered_df.date.min(), filtered_df.date.max())), 0))
    mean_posts_week = int(round(filtered_df.groupby('date_week').cnt.sum().mean(), 0))
    mean_posts_month = int(round(filtered_df.groupby('date_month').cnt.sum().mean(), 0))

    # среднее количество публикаций в день
    # среднее количество публикаций в неделю
    # среднее количество публикаций в месяц

    return mean_posts_day, mean_posts_week, mean_posts_month

#-----------------------------Метрики по просмотрам-------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

def calculate_mean_views(channel):
    filtered_df = post_view[post_view.channel_name==channel].copy()
    mean_views = int(round(filtered_df[['post_id', 'current_views']].drop_duplicates().current_views.mean(), 0))
    
    # Среднее количество просмотров одной публикации
    
    return mean_views 

#-----------------------------Метрики по реакциям-------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

def calculate_mean_reacts(channel, react1='', perc1=0, react2='', perc2=0, react3='', perc3=0):
    filtered_df = gr_pvr[gr_pvr.channel_name == channel]
    
    mean_reacts = int(round(filtered_df[['post_id', 'react_cnt_sum']].drop_duplicates().react_cnt_sum.mean(), 0))
    mean_idx = round(filtered_df[['post_id', 'idx_active']].drop_duplicates().idx_active.mean(), 1)
    
    allReact = filtered_df[filtered_df.reaction_type.apply(len)==1].react_cnt.sum()
    top3react = filtered_df[filtered_df.reaction_type.apply(len)==1].groupby('reaction_type').react_cnt.sum().reset_index()\
                                                                    .sort_values('react_cnt', ascending=False).head(3).reset_index()
    top3react.loc[:, 'react_cnt_perc'] = round(top3react.react_cnt/allReact*100, 0)
    cnt_react = top3react.shape[0]
    
    if cnt_react == 3:
        react1, perc1 = top3react.reaction_type[0], int(top3react.react_cnt_perc[0])
        react2, perc2 = top3react.reaction_type[1], int(top3react.react_cnt_perc[1])
        react3, perc3 = top3react.reaction_type[2], int(top3react.react_cnt_perc[2])
    elif cnt_react == 2:
        react1, perc1 = top3react.reaction_type[0], int(top3react.react_cnt_perc[0])
        react2, perc2 = top3react.reaction_type[1], int(top3react.react_cnt_perc[1])
    elif cnt_react == 1:
        react1, perc1 = top3react.reaction_type[0], int(top3react.react_cnt_perc[0])

    # Среднее количество реакций на публикацию
    # Средний индекс активности
    # 3 самых популярных реакий и их доли от всех реакций 

    return mean_reacts, mean_idx, react1, perc1, react2, perc2, react3, perc3



# Настройка приложения Dash

#[ "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"] 
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
    'https://fonts.googleapis.com/css?family=Merriweather|Open+Sans&display=swap',
    'Desktop/notebooks/custom-styles.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets )

#Добавляем выпадающий список для названия канала
filtr_channels = sorted(channels.channel_name.unique())

#-------------------------------------------------------------------------------------------------------------
# Макет приложения

server = app.server

# Определение стилей
styles = {
    'container': {
        'padding': '30px',
        'maxWidth': '1200px',
        'margin': '0 auto',
        'backgroundColor': '#ffb347',
        'boxShadow': '0 10px 15px rgba(0,0,0,0.05)',
        'borderRadius': '10px'
    },
    'header': {
        'backgroundColor': '#ffb347',
        'fontFamily': 'Open Sans, sans-serif', #'Merriweather, serif',
        'fontSize': '28px',
        'lineHeight': '36px',
        'color': '#333',
        'marginTop': '20px',
        'marginBottom': '5px',
        "font-weight": "bold"
    },
    'subheader_title': {
        'fontFamily': 'Open Sans, sans-serif',
        'fontSize': '16px',
        'lineHeight': '24px',
        'color': '#666',
        'marginBottom': '20px',
        "font-weight": "bold"
    },
    'subheader': {
        'fontFamily': 'Open Sans, sans-serif',
        'fontSize': '14px',
        'lineHeight': '24px',
        'color': '#666',
        'marginBottom': '10px',
    },
    'dropdown': {
        'fontFamily': 'Open Sans, sans-serif',
        'fontSize': '14px',
        'lineHeight': '21px',
        'color': '#444',
        'backgroundColor': '#ffb347',  # Фон блока
        'border': '3px solid #f5dfbf',  # Рамки блока
        'borderRadius': '14px',
        'padding': '0px 0px',
        'marginTop': '0px',
        'marginBottom': '0px'
    }
,
        'dropdown_options': {  # Дополнительные стили для опций
        'backgroundColor': '#f5dfbf',  # Фон выпадающего списка
        'color': '#444'             # Цвет текста внутри опции
    }
,
    'slider': {
        'fontFamily': 'Open Sans, sans-serif',
        'fontSize': '14px',
        'lineHeight': '21px',
        'color': '#444',
        'marginBottom': '20px',
        "trackBackgroundColor": "lightgray",  # Цвет фона дорожки ползунка
        "highlightColor": "#f5dfbf",             # Цвет выделенной области между ползунками
        "handleBorderColor": "red"       # Цвет рамки ползунков        
    },
    'graph_container': {
        'marginBottom': '40px'
        
    },
    'data_table': {
        'fontFamily': 'Open Sans, sans-serif',
        'fontSize': '12px',
        'lineHeight': '21px',
        'color': '#444',
        'borderCollapse': 'separate', #'collapse',
        'borderSpacing': '0',
        'width': '100%',
        'marginBottom': '40px'
    },
    'data_table_header': {
        'backgroundColor': '#f5dfbf', #'#eaeaea',
        'fontWeight': '600',
        'textAlign': 'left',
        'padding': '8px',
        'borderBottom': '1px solid #ddd'
    },
    'data_table_row': {
        'borderBottom': '1px solid #ddd',
        'padding': '8px'
    },
    'data_table_cell': {
        'padding': '8px',
        'textAlign': 'left',
        'border': '2px solid #ddd',
        'border-radius': '18px'
    }
    , 'buttons': {"font-size": "11px"
                  , 'margin-right': '7px'
                  , "background-color": '#ffb347' 
                  , "border-radius": "35px"
                  , "border-width": "2px"
                  , "border-color": '#f5dfbf'
                  , "box-shadow": "0px"
                  , 'color': 'black'
        
    }
    , 'metric_numbers': {
        'fontSize': '14px'
        , 'color': 'brown'
        , "font-weight": "bold"      
    }
}

# Создание компонентов фильтра для каждого столбца
filter_components = []
filter_columns_table_id = ['id']  #['id','date', 'time',  'text']
columns_table_id  = ['id','date', 'time',  'text']
label_style = {'display': 'inline-block', 'vertical-align': 'middle', 'white-space': 'nowrap'}
for col in filter_columns_table_id :
    filter_components.append(
        html.Div([
            #f"{col}: ",
            #html.Label(col, style=label_style),
            dcc.Input(id=f'input_{col}'
                      , placeholder = "Введите номер id поста"
                      , type='text'
                      , style={'width': '100%', 'margin-bottom': '10px', 'color': 'brown', "background-color": '#ffb347'})
        ])
    )
        
# Макет приложения
app.layout = html.Div([
    
    html.Div(className='container', style=styles['container'], children=[

     html.Div(className='row', style={'display': 'flex', 'margin-bottom': '40px'}, children=[
     
             html.Div(style={'width': '67%', 'height': '100%', 'marginRight': '30px'},  children=[   
                html.H1('Simulative', style=styles['header']),
                html.H2('Дашборд по анализу Telegram-каналов', style=styles['subheader_title']),
                html.Div(className='col-md-12', children=[
                        dcc.Dropdown(
                            id='channel-dropdown',
                            options=[{'label': c, 'value': c} for c in posts['channel_name'].unique()],
                            value=posts['channel_name'].unique()[0],
                            clearable=False,
                            #className = 'custom-select',
                            style=styles['dropdown']
                                    )
                        ])
              ]),
        
                 html.Div(style={'width': '27%', 'height': '100%', 'marginLeft': '30px'},  children=[
                      html.Img(id="image_wc", style={'width': '100%', 'height': '100%'})
                     ])
        ])

 , html.Div(className='row', style={'display': 'flex',  'margin-bottom': '40px'}, children=[       
   # Карточки с метриками

     # Колонка1
     html.Div(style={'width': '22%', 'height': '100%', 'marginRight': '30px'}, children=[    
                    html.Div([
                        html.Span('📈', style={'fontSize': '24px'}), 
                        html.Span('Средний ежедневный прирост  ', style={'fontSize': '12px'}),
                        html.Span(id='mean_subs_pos', style=styles['metric_numbers'])
                    ]),
                    html.Div([
                        html.Span('📉', style={'fontSize': '24px'}), 
                        html.Span('Средний ежедневный отток  ', style={'fontSize': '12px'}),
                        html.Span(id='mean_subs_neg', style=styles['metric_numbers'])
                    ]),
                    html.Div([
                        html.Span('🚀', style={'fontSize': '24px'}), 
                        html.Span('Максимальный прирост  ', style={'fontSize': '12px'}),
                        html.Span(id='max_subs_pos', style=styles['metric_numbers'])
                    ]),
                    html.Div([
                        html.Span('🆘', style={'fontSize': '24px'}), 
                        html.Span('Максимальный отток  ', style={'fontSize': '12px'}),
                        html.Span(id='max_subs_neg', style=styles['metric_numbers'])
                    ])
     ]),

    # Колонка2
    html.Div(style={'width': '22%',  'height': '100%', 'marginRight': '30px'}, children=[         
                    html.Div([
                        html.Span('📋', style={'fontSize': '24px'}), 
                        html.Span('В среднем постов в день  ', style={'fontSize': '12px'}),
                        html.Span(id='mean_posts_day', style=styles['metric_numbers'])
                    ]),
                     html.Div([
                        html.Span('📜', style={'fontSize': '24px'}), 
                        html.Span('В среднем постов в неделю  ', style={'fontSize': '12px'}),
                        html.Span(id='mean_posts_week', style=styles['metric_numbers'])
                    ]),    
                    html.Div([
                        html.Span('🗂️', style={'fontSize': '24px'}), 
                        html.Span('В среднем постов в месяц  ', style={'fontSize': '12px'}),
                        html.Span(id='mean_posts_month', style=styles['metric_numbers'])
                    ]),        
     ]),   

    # Колонка3
    html.Div(style={'width': '22%', 'height': '100%', 'marginRight': '30px'}, children=[      
                    html.Div([
                        html.Span('👀', style={'fontSize': '24px'}), 
                        html.Span('В среднем просмотров  ', style={'fontSize': '12px'}),
                        html.Span(id='mean_views', style=styles['metric_numbers'])
                    ]),   

 
                    html.Div([
                        html.Span('🐾', style={'fontSize': '24px'}), 
                        html.Span('В среднем реакций  ', style={'fontSize': '12px'}),
                        html.Span(id='mean_reacts', style=styles['metric_numbers'])
                    ]),   

                    html.Div([
                        html.Span('💎', style={'fontSize': '24px'}), 
                        html.Span('В среднем уровень активности  ', style={'fontSize': '12px'}),
                        html.Span(id='mean_idx', style=styles['metric_numbers'])
                    ]),   
      ]),

    # Колонка4
    html.Div(style={'width': '22%', 'height': '100%', 'marginLeft': '30px'}, children=[         
                    html.Div([
                        html.Span('🥇', style={'fontSize': '24px'}),
                        html.Span('  Доля реакции ', style={'fontSize': '12px'}),
                        html.Span(id='react1', style={'fontSize': '24px'}),
                        html.Span(':  ', style={'fontSize': '12px'}),
                        html.Span(id='perc1', style=styles['metric_numbers'])
                    ]),   
                    html.Div([
                        html.Span('🥈', style={'fontSize': '24px'}),
                        html.Span('  Доля реакции ', style={'fontSize': '12px'}),
                        html.Span(id='react2', style={'fontSize': '24px'}),
                        html.Span(':  ', style={'fontSize': '12px'}),
                        html.Span(id='perc2', style=styles['metric_numbers'])
                    ]), 
                    html.Div([
                        html.Span('🥉', style={'fontSize': '24px'}),
                        html.Span('  Доля реакции ', style={'fontSize': '12px'}),
                        html.Span(id='react3', style={'fontSize': '24px'}),
                        html.Span(':   ', style={'fontSize': '12px'}),
                        html.Span(id='perc3', style=styles['metric_numbers'])
                    ]),         
    ])   
     
 ])
    
   , html.Div(className='row', style={'display': 'flex', 'margin-bottom': '40px'}, children=[
        
            # Правая колонка 
            html.Div(style={'width': '47%', 'height': '100%', 'marginRight': '30px'},  children=[
                html.Div(className='row', children=[
                    html.Div(className='col-md-12', style=styles['graph_container'], children=[  
                        html.H4("Аудитория на момент измерения", style=styles['subheader_title']),
                         html.P("График показывает изменение общего количества подписчиков с течением времени. Он помогает отслеживать динамику роста аудитории и выявлять периоды активного притока или оттока подписчиков. Анализ графика позволяет корректировать стратегию продвижения и создавать контент, который привлечет и удержит больше подписчиков (Процентные значения индикаторов указывают на изменения по сравнению с предыдущими аналогичными периодами).", style=styles['subheader']),
                                               
                        dcc.Graph(id='graph2')
                    ]),

                    html.Div(className='col-md-12',  style={'marginBottom': '40px'}, children=[
                        html.H4("Динамика подписок", style=styles['subheader_title']),
                        html.P("Этот график показывает два ключевых показателя: количество пользователей, которые подписались на канал, и тех, кто отписался. Он помогает отслеживать, насколько эффективно ваш контент привлекает новую аудиторию и удерживает существующую. Анализируя этот график, можно сделать выводы о том, какие периоды были наиболее успешными в привлечении подписчиков, а также выявить моменты, когда наблюдалось значительное снижение аудитории. Этот анализ позволит вам скорректировать стратегию создания контента и время его публикации для достижения лучших результатов.", style=styles['subheader'])
                        , dcc.Graph(id='graph-with-slider')
                    ]),


                    html.Div(className='col-md-12', style={'marginBottom': '40px', 'marginTop': '0px'}, children=[
                        dcc.RangeSlider(
                            id='date-slider',
                            min=0,
                            max=(subs['datetime'].max() - subs['datetime'].min()).total_seconds(),
                            value=[0, (subs['datetime'].max() - subs['datetime'].min()).total_seconds()],
                            marks={
                                int((date - subs['datetime'].min()).total_seconds()): {
                                    'label': date.strftime("%b %d, %H:%M"),
                                    'style': {'fontSize': '12px'}
                                } for date in subs['datetime'][::len(subs) // 5]
                            },
                            step=None,
                            updatemode='drag'
                        )
                    ]),

                    html.Div(className='col-md-12',  style={'marginBottom': '40px'}, children=[
                        html.H4("Визуализация интереса к контенту", style=styles['subheader_title']),
                        html.P("Ось Y здесь показывает, насколько активно аудитория реагирует на ваш контент, а ось X – сколько раз этот контент просмотрен. Чем крупнее пузырек, тем больше реакций собрал пост. Если пузырёк высоко взлетел, значит тема 'зашла' – люди не только смотрят, но и активно реагируют. А вот маленькие и низко расположенные пузырьки подсказывают, что стоит задуматься над изменениями. Этот график поможет вам понять, какие темы цепляют аудиторию, когда лучше всего публиковать новые материалы и как улучшить те посты, которые пока не так популярны.", style=styles['subheader'])
                        #, dcc.Graph(id='graph-with-slider')
                    ]),                    


                    html.Div(className='col-md-12', style=styles['graph_container'], children=[  
                        #html.H4("Аудитория на момент измерения", style=styles['subheader_title']),
                         #html.P("График показывает изменение общего количества подписчиков с течением времени. Он помогает отслеживать динамику роста аудитории и выявлять периоды активного притока или оттока подписчиков. Анализ графика позволяет корректировать стратегию продвижения и создавать контент, который привлечет и удержит больше подписчиков (Процентные значения индикаторов указывают на изменения по сравнению с предыдущими аналогичными периодами).", style=styles['subheader']),
                         html.Div(
                           # className='d-flex justify-content-end mb-2',  # Bootstrap классы для выравнивания по правому краю
                            children=[
                                html.Button("3д", id="btn-3d_2", n_clicks=0, style=styles['buttons'], className="btn btn-primary btn-flat"),
                                html.Button("1н", id="btn-1w_2", n_clicks=0, style=styles['buttons'], className="btn btn-primary btn-flat"),
                                html.Button("1м", id="btn-1m_2", n_clicks=0, style=styles['buttons'], className="btn btn-primary btn-flat"),
                                html.Button("All (6м)", id="btn-all_2", n_clicks=0, style=styles['buttons'], className="btn btn-primary btn-flat")
                            ]
                        )                                               
                        , dcc.Graph(id='graph6')
                    ]),
                    
                ])
            ]),

            # Левая колонка с графиками
            html.Div(style={'width': '47%', 'height': '100%', 'marginLeft': '30px'}, children=[
                     
                html.Div(className='row', children=[
                    html.Div(className='col-md-12', style=styles['graph_container'], children=[
                        html.H4("Суточные показатели публикаций", style=styles['subheader_title']),
                         html.P("График показывает количество публикаций конкурента. Процентные значения  за разные периоды (день, неделя и месяц) указывают на изменения активности по сравнению с предыдущими аналогичными периодами. Анализ этих данных поможет понять, как часто и интенсивно конкурент публикует материалы, что может быть полезным для корректировки вашей собственной стратегии создания контента.", style=styles['subheader']),
                        dcc.Graph(id='graph1')
                    ]),
                    
                    html.Div(className='col-md-12', children=[
                        html.H4("График публикаций", style=styles['subheader_title']),
                        html.P("Этот график является полезным инструментом для понимания того, когда ваши конкуренты выпускают контент или если вы планируете протестировать новый график публикации своих постов (учитываются последние шесть месяцев).", style=styles['subheader']),
                           # Контейнер для кнопок
                        html.Div(
                           # className='d-flex justify-content-end mb-2',  # Bootstrap классы для выравнивания по правому краю
                            children=[
                                html.Button("3д", id="btn-3d", n_clicks=0, style=styles['buttons'], className="btn btn-primary btn-flat"),
                                html.Button("1н", id="btn-1w", n_clicks=0, style=styles['buttons'], className="btn btn-primary btn-flat"),
                                html.Button("1м", id="btn-1m", n_clicks=0, style=styles['buttons'], className="btn btn-primary btn-flat"),
                                html.Button("All (6м)", id="btn-all", n_clicks=0, style=styles['buttons'], className="btn btn-primary btn-flat")
                            ]
                        )
                        
                        
                        , dcc.Graph(id='graph3')


                    ]),

                    html.Div(className='col-md-12', style={'overflow-y': 'auto', 'max-height': '870px', 'margin-left': '20px'}, children=[
                        html.H4("Динамика просмотров по дням", style=styles['subheader_title']),
                        html.P("Эта таблица помогает определить оптимальное время для публикаций: если в первые сутки после публикации она собирает более 35% всех просмотров, это успешное время публикации; иначе стоит пересмотреть график размещения контента, чтобы новые публикации не затерялись среди конкурентов. Также можно обнаружить возможную мошенническую активность: например, если за одни сутки видео набирает 80% общего количества просмотров, следует проявить осторожность, проанализировать частоту подобных аномалий и сделать выводы (проценты приведены, как пример).", style=styles['subheader'])
                        
                        , dcc.Slider(
                            id='hours-slider',
                            min=0,
                            max=72,
                            step=1,
                            value=5,
                            marks={i: str(i) + 'д' for i in range(1, 73, 4)} 
                            , className='my-custom-slider' 
                        ),
                        html.Table(id='table-container', style=styles['data_table'], children=[
                            html.Thead(children=[
                                html.Tr(children=[
                                    html.Th('ID поста и дата', style=styles['data_table_header']),
                                    html.Th('Текущие просмотры', style=styles['data_table_header']),
                                    *[html.Th(f'{i}д', style=styles['data_table_header']) for i in range(1, 25)]
                                ])
                            ]),
                            html.Tbody(id='table-body', children=[])
                        ])
                    ])

                    , html.Div(className='col-md-12', style={'marginTop': '50px'}, children=[
                        html.H4("Просмотр текста поста и даты по номеру ID: ", style=styles['subheader']),
                        # Фильтры
                        *filter_components,
                                # Таблица
                                html.Br(),
                                html.Table(id='table_id')                        
                    ]),
                ])
            ]),
        
        ])
    ])
], style={'font-family': 'Open Sans, sans-serif'})


#----------------------------------------------------------------------------------------------------------------------------
#-----------------------------Метрики----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------


# Обновление метрик при выборе канала
@app.callback(
    [
        Output('mean_subs_pos', 'children'),
        Output('mean_subs_neg', 'children'),
        Output('max_subs_pos', 'children'),
        Output('max_subs_neg', 'children'),
        Output('mean_posts_day', 'children'),
        Output('mean_posts_week', 'children'),
        Output('mean_posts_month', 'children'),
        Output('mean_views', 'children'),
        Output('mean_reacts', 'children'),
        Output('mean_idx', 'children'),
        Output('react1', 'children'),
        Output('perc1', 'children'),
        Output('react2', 'children'),
        Output('perc2', 'children'),
        Output('react3', 'children'),
        Output('perc3', 'children')
        
    ],
    Input('channel-dropdown', 'value'))

def update_metrics(channel):
    mean_subs_pos, mean_subs_neg, max_subs_pos, max_subs_neg = calculate_mean_max_subs(channel)
    mean_posts_day, mean_posts_week, mean_posts_month = calculate_mean_posts(channel)
    mean_views  = calculate_mean_views(channel)
    mean_reacts, mean_idx, react1, perc1, react2, perc2, react3, perc3 = calculate_mean_reacts(channel)
    
    return str(mean_subs_pos), str(mean_subs_neg), str(max_subs_pos), str(max_subs_neg), str(mean_posts_day),\
    str(mean_posts_week), str(mean_posts_month), str(mean_views),\
    str(mean_reacts), f"{mean_idx}%", str(react1), f"{perc1}%", str(react2), f"{perc2}%", str(react3), f"{perc3}%" 


#----------------------------------------------------------------------------------------------------------------------------
#-----------------------------Publications-----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

@app.callback(Output('graph1', 'figure'), [Input('channel-dropdown', 'value')])
def update_graph1(channel):
    #filtered_df = subdf.query(f"country=='{country}'")
    subdf = posts[posts.channel_name == channel][['channel_name', 'date', 'cnt']].drop_duplicates()

    # Создание subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"rowspan": 3}, {'type': 'indicator'}],
            [None, {'type': 'indicator'}],
            [None, {'type': 'indicator'}],
      ],
        vertical_spacing=0.08
    )
    
    mean_cnt = subdf.cnt.mean()
    #colors = ['#89cff0' if val > 3*mean_cnt else '#7F7F7F' for val in subdf['cnt']]  # Легкий оттенок коричневого для больших значений
    colors = ['#8B4513' if val >= 2*mean_cnt else '#F5DEB3' for val in subdf['cnt']]  # '#f5dfbf'

    
    fig.add_trace(go.Bar(x = subdf.date, y=subdf.cnt, marker_color=colors,
                        hovertemplate='%{x} <br>Публикаций: %{y}<extra></extra>'), row=1, col=1)
    period_names = dict({'days':'вчера', 'weeks': 'неделю', 'months': 'месяц'})
    
    for i, period in enumerate([('days', 'days', 1), ('weeks', 'weeks', 1), ('months', 'months', 1)]):
        current = subdf[(subdf.date <= date_ago(period[0])) & (subdf.date > date_ago(period[1], period[2]))].cnt.sum()
        previous = subdf[(subdf.date <= date_ago(period[1], period[2])) & (subdf.date > date_ago(period[1], period[2]*2))].cnt.sum()
            
        fig.add_trace(
                go.Indicator(
                    value=current,
                    title={"text": f"<span style='font-size:0.8em;color:gray'>Публикаций за {period_names[period[0]]}</span>"},
                    mode="number+delta",
                    delta={'reference': previous, 'relative': True, "valueformat": ".2%"},
                ), row=i+1, col=2
            )
    
    # Настройки стиля
    fig.update_layout(
       # title_text=f"Суточные показатели публикаций для канала: {channel}",
        template="simple_white",
        font_family="Georgia",
        font_size=12,
        margin=dict(l=40, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            rangeselector=dict(  # Добавляем элементы управления диапазоном
                bgcolor= '#f5dfbf' ,  # Фоновый цвет области с кнопками
                font=dict(color="#333"),  # Цвет текста на кнопках
                activecolor= '#ffb347',  # Цвет активной кнопки
                bordercolor='#f5dfbf',  # Цвет рамки вокруг кнопок                     
                buttons=list([
                    dict(count=2, label="2д", step="day", stepmode="backward"),
                    dict(count=14, label="2н", step="day", stepmode="backward"),
                    dict(count=2, label="2м", step="month", stepmode="backward"),
                    dict(step="all")  # Кнопка для просмотра всего диапазона
                ])
            )  ) 
    )
    return fig

#----------------------------------------------------------------------------------------------------------------------------
#-----------------------------SUBS-------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
@app.callback(Output('graph2', 'figure'), [Input('channel-dropdown', 'value')])
def update_graph1(channel):
    
    subdf = subs[subs.channel_name == channel][['channel_name', 'date'
                                                      ,'subs_cnt', 'subs_change', 'datetime']].drop_duplicates()
    subdf.sort_values(by=['channel_name', 'datetime'], inplace=True)
    
    # Создание subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [ {"rowspan": 3}, {'type': 'indicator'}],
            [None, {'type': 'indicator'}],
            [ None, {'type': 'indicator'}],
      ],
        vertical_spacing=0.08
    )
    
    fig.add_trace(
        go.Scatter(
            x=subdf.datetime,
            y=subdf.subs_cnt,
            fill='tozeroy',
            mode='lines+markers',
            line_color= '#f5dfbf', #'#7F7F7F',
            marker_color='#f5dfbf', #'#7F7F7F',
            marker_line_color='#f5dfbf', #'#7F7F7F',
            marker_line_width=1,
            marker_size=5,
            showlegend=False,
            hovertemplate='%{x}  <br>Подписчиков вс: %{y}<extra></extra>'
        ),
        row=1,
        col=1
    )
        
    period_names = dict({'days':'вчера', 'weeks': 'неделю', 'months': 'месяц'})   
    for i, period in enumerate([('days', 'days', 1), ('weeks', 'weeks', 1), ('months', 'months', 1)]):
        subdf.sort_values(by='date', inplace=True)
        current = subdf[subdf.date <= date_ago(period[0])].subs_change.sum() - subdf[
                                                                            subdf.date <= date_ago(period[1], period[2])].subs_change.sum()
        previous = subdf[subdf.date <= date_ago(period[1], period[2])].subs_change.sum() - subdf[
                                                                            subdf.date <= date_ago(period[1], period[2]*2)].subs_change.sum()
        
        fig.add_trace(
            go.Indicator(
                value=current,
                title={"text": f"<span style='font-size:0.8em;color:gray'>Подписчиков за {period_names[period[0]]}</span>"},
                mode="number+delta",
                delta={'reference': previous, 'relative': True, "valueformat": ".2%"},
            ), row=i+1, col=2
        )


    # Настройки стиля
    fig.update_layout(
        #title_text=f"GDP per Capita over Time for {country}",
        template="simple_white",
        font_family="Georgia",
        font_size=12,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            rangeselector=dict(  # Добавляем элементы управления диапазоном
                bgcolor= '#f5dfbf' ,  # Фоновый цвет области с кнопками
                font=dict(color="#333"),  # Цвет текста на кнопках
                activecolor= '#ffb347',  # Цвет активной кнопки
                bordercolor='#f5dfbf',  # Цвет рамки вокруг кнопок                    
                buttons=list([
                    dict(count=2, label="2д", step="day", stepmode="backward"),
                    dict(count=14, label="2н", step="day", stepmode="backward"),
                    dict(count=2, label="2м", step="month", stepmode="backward"),
                    dict(step="all")  # Кнопка для просмотра всего диапазона
                ])
            )  ) 
    )
    return fig

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

# Модифицируем существующий callback
@app.callback(
    Output('graph3', 'figure'),
    Input('channel-dropdown', 'value'),
    Input("btn-3d", "n_clicks"),
    Input("btn-1w", "n_clicks"),
    Input("btn-1m", "n_clicks"),
    Input("btn-all", "n_clicks")
)
def update_graph3(channel, btn_3d_n_clicks, btn_1w_n_clicks, btn_1m_n_clicks, btn_all_n_clicks):
    if channel is None:
        return {}
        
    # Получаем контекст вызова
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Фильтрация данных в зависимости от нажатой кнопки
    if button_id == "btn-3d":
        filtered_df = posts[(posts.channel_name == channel)&(posts.date>=date_ago('days', 2))]
    elif button_id == "btn-1w":
        filtered_df = posts[(posts.channel_name == channel)&(posts.date>=date_ago('weeks', 1))]
    elif button_id == "btn-1m":
        filtered_df = posts[(posts.channel_name == channel)&(posts.date>=date_ago('months', 1))]
    else:
        filtered_df = posts[(posts.channel_name == channel)&(posts.date>=date_ago('months', 6))]

    # Генерация данных
    filtered_df = filtered_df[['date', 'hour', 'cnt']].rename(columns={'cnt': 'publications'}).sort_values('date')
    raw_index = filtered_df.set_index(['date', 'hour'])
    
    dates = pd.to_datetime(filtered_df.date).unique().tolist()
    index = pd.MultiIndex.from_product([filtered_df.date.unique(), range(1, 25)], names=['date', 'hour'])
    raw = pd.DataFrame(index=index)
    df = raw.merge(raw_index, left_index=True, right_index=True, how='left')
    df.fillna(0, inplace=True)
    df = df.reset_index().drop_duplicates(subset=['date', 'hour']).set_index(['date', 'hour'])
    
    # Преобразование данных в формат, подходящий для heatmap
    z_values = df['publications'].unstack(level=-1)
    x_labels = [str(hour) for hour in range(1,25)]
    y_labels = [date.strftime('%Y-%m-%d') for date in dates]  
    
    fig = go.Figure(
        data=[
           go.Heatmap(
                    z= pd.DataFrame([[1] * len(x_labels)]*len(y_labels), columns=range(1,25), index=y_labels), #[[1] * len(x_labels)] * len(y_labels),  # Матрица одинаковых значений для всех ячеек
                    
                    x=x_labels,
                    y=y_labels,
                    colorscale=[[0, '#ffb347'], [1, '#ffb347']],  # Градиент от белого к темно-синему
                    showscale=False,
                    hovertemplate='%{y} <br>%{x} ч <br>Публикаций: %{z}<extra></extra>'
                ),
                go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                colorscale=[[0, '#F5DEB3'], [1, "#006a4e"]],  # [[0, "#e5e4e2"], [1, "#006a4e"]], Серый цвет для фона
                showscale=False,
                xgap=10,  # Зазор между ячейками по горизонтали
                ygap=10,   # Зазор между ячейками по вертикали
                hovertemplate='%{y} <br>%{x} ч <br>Публикаций: %{z}<extra></extra>'
            )
        ],
    ).update_layout(
        font_family='Arial',
    
        margin=dict(l=30, r=50, t=50, b=20),
        paper_bgcolor='#ffb347',
        plot_bgcolor='#ffb347',
        legend_title_font_color="#212121",
        legend_font_color="#212121",
        legend_borderwidth=0,
        hoverlabel_font_family='Arial',
        hoverlabel_font_size=12,
        hoverlabel_font_color='#212121',
        hoverlabel_align='auto',
        hoverlabel_namelength=-1,
        hoverlabel_bgcolor='#FAFAFA',
        hoverlabel_bordercolor='#E5E4E2'
        
    )

    # Ограничиваем количество меток на оси Y до 10
    if len(y_labels) > 10:
        y_labels_subset = y_labels[::max(len(y_labels)//10,1)]
    else:
        y_labels_subset = y_labels
    
    # Перемещение подписей часов наверх
    fig.update_xaxes(side="top", tickfont=dict(family='Arial', size=12), title_font=dict(family='Arial', size=14))
    
    fig.update_yaxes(
        autorange="reversed",
        #dtick=max(len(y_labels) // 10, 1),
        ticktext=y_labels,
        #tickvals= [datetime.datetime.strptime(date, "%Y-%m-%d").timestamp() for date in y_labels_subset],
        tickformat="%b %d, %y",
        tickfont={"family": "Arial", "size": 8},  # Уменьшаем размер шрифта для компактности
        title_font={"family": "Arial", "size": 14}
    )
    
    # Добавляем полосу прокрутки для оси Y
    fig.update_layout(
        font_size=9,
        yaxis_title="Дата",
        xaxis_title="Часы",    
        yaxis=dict(
            autorange="reversed",
             # tickangle=45,  # Наклон меток для улучшения читаемости
 ) 
    )    
    return fig
#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
        
@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('channel-dropdown', 'value'),
    Input('date-slider', 'value'))
def update_graph4(channel, slider_range):
    if channel is None or slider_range is None:
        return {}
        
    subdf_channel = subs[subs['channel_name'] == channel]
    
    # Проверяем, что дата присутствует и не пуста
    if len(subdf_channel) == 0 or 'datetime' not in subdf_channel.columns:
        return {}    
    # Преобразуем строку в datetime
    subdf_channel.loc[:, 'datetime'] = pd.to_datetime(subdf_channel['datetime'])
    start_time = subdf_channel['datetime'].min() + pd.Timedelta(seconds=slider_range[0])
    end_time = subdf_channel['datetime'].min() + pd.Timedelta(seconds=slider_range[1])

    filtered_df = subdf_channel[(subdf_channel['datetime'] >= start_time) & (subdf_channel['datetime'] <= end_time)]
    
    filtered_df_uniq = filtered_df[['date', 'day_change_pos', 'day_change_neg']].drop_duplicates()
    
    #colors = [ '#A9A9A9' if val < 0 else  '#89cff0' for val in filtered_df['subs_change']]
    #colors = [ '#8B0000' if val < 0 else  '#F5DEB3' for val in filtered_df_uniq['subs_change']]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=filtered_df_uniq['date'], y=filtered_df_uniq['day_change_pos'], marker_color='#F5DEB3', hovertemplate='%{x} <br>Подписались: %{y} <extra></extra>'))
    fig.add_trace(go.Bar(x=filtered_df_uniq['date'], y=filtered_df_uniq['day_change_neg'], marker_color='#8B0000', hovertemplate='%{x} <br>Отписались: %{y}<extra></extra>'))

    fig.update_layout(
        showlegend=False,
        paper_bgcolor= '#ffb347', #'#FFFFFF',
        plot_bgcolor=  '#ffb347', #'#FFFFFF',
        font_family='Georgia',
        title_font_size=24,
        title_x=0.5,
        margin=dict(l=40, r=60, t=40, b=10),
        yaxis_title="Изменение подписок",
        xaxis_title="Дата и время",
        xaxis=dict(
            rangeselector=dict(  # Добавляем элементы управления диапазоном
                bgcolor= '#f5dfbf' ,  # Фоновый цвет области с кнопками
                font=dict(color="#333"),  # Цвет текста на кнопках
                activecolor= '#ffb347',  # Цвет активной кнопки
                bordercolor='#f5dfbf',  # Цвет рамки вокруг кнопок                
                buttons=list([
                    dict(count=3, label="3д", step="day", stepmode="backward"),
                    dict(count=7, label="1н", step="day", stepmode="backward"),
                    dict(count=1, label="1м", step="month", stepmode="backward"),
                    dict(step="all")  # Кнопка для просмотра всего диапазона
                ])
            )  ) 
    )
    return fig


@app.callback(
    Output('date-slider', 'marks'),
    Input('channel-dropdown', 'value'))
def update_slider_marks(channel):
    if channel is None:
        return {}

    subdf_channel = subs[subs['channel_name'] == channel]
    dates = sorted(subdf_channel.date)
    # Преобразуем список строк в список дат
    dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]
    date_min = min(dates)
    if len(dates) > 0:
        marks = {
            int(pd.Timedelta(date - date_min).total_seconds()): {
                'label': date.strftime("%b %d"), #format_date(date, "MMM d", locale='ru_RU').title()
                'style': {
                    'fontSize': '12px',
                    'color': 'black',
                    'backgroundColor': '#f5dfbf', #'white',
                    'borderRadius': '5px',
                    'padding': '2px',
                    'display': 'block',
                    'width': 'auto',
                    'transform': 'translateX(-50%)'
                }
            } for date in dates[::len(dates)//6]
        }
    else:
        marks = {}
    return marks
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output("table-container", "children"),
    Input("hours-slider", "value"),
     Input("channel-dropdown", "value") 
)
def update_table(max_days, channel):
    return create_table(post_view, max_days, channel)
    
#-------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------    


import colorlover as cl

# Модифицируем существующий callback
@app.callback(
    Output('graph6', 'figure'),
    Input('channel-dropdown', 'value'),
    Input("btn-3d_2", "n_clicks"),
    Input("btn-1w_2", "n_clicks"),
    Input("btn-1m_2", "n_clicks"),
    Input("btn-all_2", "n_clicks")
)
def update_graph6(channel, btn_3d_n_clicks, btn_1w_n_clicks, btn_1m_n_clicks, btn_all_n_clicks):
    if channel is None:
        return {}
        
    # Получаем контекст вызова
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    
    # Фильтрация данных в зависимости от нажатой кнопки
    def buttons_cond(df, channel, button_id):
        if button_id == "btn-3d_2":
            filtered_df = df[(df.channel_name == channel)&(df.post_datetime.str[:10]>=date_ago('days', 2))]
        elif button_id == "btn-1w_2":
            filtered_df = df[(df.channel_name == channel)&(df.post_datetime.str[:10]>=date_ago('weeks', 1))]
        elif button_id == "btn-1m_2":
            filtered_df = df[(df.channel_name == channel)&(df.post_datetime.str[:10]>=date_ago('months', 1))]
        else:
            filtered_df = df[(df.channel_name == channel)&(df.post_datetime.str[:10]>=date_ago('months', 6))]
            
        return filtered_df

    # Генерация данных
    filtered_gr_pvr = buttons_cond(gr_pvr, channel, button_id)
    #table
    gr_pvr_sum = filtered_gr_pvr.drop(['reaction_type', 'react_cnt'], axis=1).drop_duplicates()

    if gr_pvr_sum.shape[0] == 0:
        return {}
    
    # Создаем градиент 
    colors = cl.scales['9']['seq']['OrRd'][::-1] 
    
# Предположим, что у тебя уже есть DataFrame под названием gr_pvr_sum
    fig = go.Figure()
    
    # Добавление точек на график
    fig.add_trace(go.Scatter(
        x=gr_pvr_sum['current_views'],
        y=gr_pvr_sum['idx_active'],
        mode='markers',
        marker=dict(
            size=gr_pvr_sum['react_cnt_sum'],
            color=gr_pvr_sum['current_views'],
            colorscale=colors,
            showscale=False,  # Скрывает colorbar
            sizemode='area',
            sizeref=2. * max(0, max(gr_pvr_sum['react_cnt_sum'])) / (18.**2),
            sizemin=4
        ),
        text=gr_pvr_sum[['post_id']],  # Показывает post_id и дату при наведении
        hoverinfo='text+x+y+z',  # Настройка информации во всплывающей подсказке
        hovertemplate=
            '<b>ID Поста:</b> %{text}<br>' +
            '<b>Текущие Просмотры:</b> %{x}<br>' +
            '<b>Количество реакций:</b> %{marker.size}<br>' +  # Добавлен размер пузыря
            '<b>Активность:</b> %{y} %<extra></extra>'
    ))
    
    # Логарифмическая ось X
    fig.update_xaxes(type="log")


    # Скрыть colorbar
    fig.update_layout(coloraxis_showscale=False)

    fig.update_layout(    
        yaxis_title="Индекс активности, %",
        xaxis_title="Текущее количество просмотров",         
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgb(102, 102, 102)',
            tickfont_color='rgb(102, 102, 102)',
            showticklabels=True,
            #dtick=10,
            ticks='outside',
            tickcolor='rgb(102, 102, 102)',
        ),
        margin=dict(l=40, r=60, t=10, b=10),
        showlegend=False,
        paper_bgcolor='#ffb347',
        plot_bgcolor='#ffb347',
        hovermode='closest',
    )
    return fig

@app.callback(Output('image_wc', 'src'), Input('channel-dropdown', 'value'))
def update_graph7(channel):
        
    posts_channel = posts[posts['channel_name'] == channel]


    words = posts_channel.text.apply(lambda t: list(set([w.lower() for w in nltk.word_tokenize(t)])- puncts - dell_words)).tolist()
    df_words = pd.DataFrame(Counter(sum(words, [])).most_common(50), columns = ['word', 'count'])
        
    def plot_wordcloud(data):
        d = {a: x for a, x in data.values}
        wc = WordCloud(background_color='#f5dfbf', color_func=gradient_color_func) #, width=480, height=360
        wc.fit_words(d)
        return wc.to_image()
            
    def make_image():
        img = BytesIO()
        plot_wordcloud(data=df_words).save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    return make_image()


@app.callback(
    Output('table_id', 'children'),
    [Input(f'input_{col}', 'value') for col in filter_columns_table_id]
)

def update_table(*args):
    # Проверка наличия введённых значений
    if any(value is not None and value != '' for value in args):
        # Получаем текущие значения фильтров
        filters = dict(zip(filter_columns_table_id, args))
        
        # Создаем маску для фильтрации данных
        mask = pd.Series(True, index=posts.index)  # Начальная маска
        for col, value in filters.items():
            if value is not None and value != '':
                try:
                    # Преобразуем значение в число, если возможно
                    numeric_value = float(value)
                    
                    # Если столбец содержит числа, применяем числовое сравнение
                    if pd.api.types.is_numeric_dtype(posts[col]):
                        mask &= (posts[col] == numeric_value)
                    else:
                        # Иначе используем текстовое сравнение
                        mask &= (posts[col].astype(str).str.contains(value))
                except ValueError:
                    # Если преобразование в число невозможно, используем текстовое сравнение
                    mask &= (posts[col].astype(str).str.contains(value))
                
        # Применяем маску к данным
        filtered_df = posts[columns_table_id][mask]
        
        # Формируем таблицу
        table_rows = [
            html.Tr([html.Th(col) for col in filtered_df.columns]),
            *[
                html.Tr([html.Td(cell, style={'vertical-align': 'top','padding': '8px'}) for cell in row])
                for _, row in filtered_df.iterrows()
            ]
        ]
        
        return table_rows
    else:
        return []  # Возвращаем пустую таблицу, если нет введённых значений



if __name__ == '__main__':
    app.run_server(debug=True, port=8016)


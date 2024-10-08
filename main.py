# %%

import json
import codecs
import copy
import re
import textwrap
import os
import pandas as pd
# from datetime import datetime
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import seaborn as sns


# path_to_file = 'Relatos 2022 sample.txt'
path_to_file = 'Reports.txt'
# path_to_file = 'test_data.txt'

curr_path = os.path.dirname(os.path.realpath(__file__))
path_to_file = os.path.join(curr_path, path_to_file)

with open(path_to_file, encoding='utf-8') as f:
    content = f.readlines()

# %%

sep = '|========================================================='
text = ''.join(content)
text = text.replace('\ufeff', '\n')
text = text.replace('\n\n', '\n')
texts = text.split(sep)
texts = [t for t in texts if len(t) > 1]
texts

# %%

class Report():

    def __init__(self, report_str) -> None:
        self.raw_report_str = report_str
        self.have_updates = False
        self.n_updates = 0
        self.title = None
        self.text_list = []
        self.date_list = []
        self.content = []
        self.main_cites = []
        self.updates_cites = []
        self.valid = False
        self.report_content = self.process()

    def process(self):
        try:
            self._process()
        except Exception as e:
            print(f"Error processing report: {e}")
            self.valid = False

    def _process(self):
        report_str = copy.copy(self.raw_report_str)
        blocks = self.parse_blocks(report_str)
        self.header_list, self.text_list = self.separate_header_from_text(blocks)
        self.parse_header()

        # a, b = self._validate_report()
        if len(blocks) > 1:
            self.have_updates = True
            self.n_updates = len(blocks) - 1
        self.find_citations()
        self.calculate_length()
        # self.organize_contents()
        self.valid = True

    def separate_header_from_text(self, blocks):
        header_list, text_list = [], []
        for block in blocks:
            header_list.append(block.split('\n')[0])
            text_list.append(''.join(block.split('\n')[1:]))
        return header_list, text_list

    def parse_header(self):
        self.string_dates = [' '.join(h.split(' ')[1:3]) for h in self.header_list]
        self.titles = [' '.join(h.split(' ')[3:]) for h in self.header_list]
        self.dates = [datetime.datetime.strptime(d, '%Y%m%d %Hh%M') for d in self.string_dates]

        self.years = [d.year for d in self.dates]
        self.months = [d.month for d in self.dates]
        self.days = [d.day for d in self.dates]
        self.hours = [d.hour for d in self.dates]
        self.minutes = [d.minute for d in self.dates]

        fdates = []
        for (d,m,y,h,M) in zip (self.days, self.months, self.years, self.hours, self.minutes):
            fdates.append(str(d) + '/' + str(m) + '/' + str(y) + ' ' + str(h) + 'h' + str(M))
        self.fdates = fdates

        # Auxiliary information
        self.is_update = [True for _ in self.dates]
        self.is_update[0] = False
        self.update_of = [self.dates[0] for _ in self.dates]
        self.update_of[0] = None

    # def remove_empty_strings(self, report_list):
    #     new_report_list = [_ for _ in report_list if len(_) > 0]
    #     return new_report_list

    def find_citations(self):
        pattern = r"report+ +\d{8}"
        self.citations = []
        for text in self.text_list:
            citations = re.findall(pattern, text, flags=re.IGNORECASE)
            self.citations.append(citations)

    def calculate_length(self):
        self.text_length = []
        for text in self.text_list:
            self.text_length.append(len(text))

    def parse_blocks(self, report_text):
        pattern = re.compile(r'\|(?:I).*?(?=\||$)', re.DOTALL)
        initiated_block = re.findall(pattern, report_text)

        pattern = re.compile(r'\|(?:A).*?(?=\||$)', re.DOTALL)
        updated_block = re.findall(pattern, report_text)

        # pattern = re.compile(r'\|(?:T).*?(?=\||$)', re.DOTALL) # self.tag_block = re.findall(pattern, report_text)
        return initiated_block + updated_block

    @property
    def citation_line(self):
        if len(self.main_cites) == 0:
            main_cites = 'Main report have no citations.'
        else:
            main_cites = f"Main report cites {', '.join(self.main_cites)}."

        if not self.have_updates:
            updates_cites = ''
        elif len(self.updates_cites) == 0:
            updates_cites = 'Updates have no citations.'
        else:
            updates_cites = f"Updates cites {', '.join(self.main_cites)}."
        return main_cites + ' ' + updates_cites

    def _validate_report(self):
        print("VALIDATING")
        valid = True
        output_ls = []
        # self.have_updates = False
        # self.n_updates = 0
        if len(self.initiated_date) == 0:
            valid = False
            output_ls.append('Error parsing initiated date.')
        if len(self.text_list) != len(self.date_list):
            valid = False
            output_ls.append('Difference between number of dates and texts.')
        return valid, '. '.join(output_ls)

    # def organize_contents(self):
    #     content = []
    #     for date, text, hoursminute, title in zip(self.dates, self.texts, self.hoursminutes, self.titles):
    #         content.append({
    #             "date": date,
    #             "time": hoursminute,
    #             "title": title,
    #             "text": text
    #         })
    #     self.content = content

    def create_repr_string(self) -> str:
        title = f"'{self.titles[0]}'"
        opening_line = f"Report {title} initiated at {self.fdates[0]}, with {self.n_updates} updates."
        update_date_line = ''
        if self.have_updates:
            update_date_line = f"Updates were made at {', '.join(self.fdates[1:])}."
        main_content = f"\nMain from {self.fdates[0]}:\n{self.text_list[0]}"
        if self.have_updates:
            for i in range(1, len(self.text_list)):
                main_content += f"\n\nUpdate from {self.fdates[i]}:\n{self.text_list[i]}"
        lines_list = [opening_line, update_date_line, self.citation_line, main_content]
        return '\n'.join(lines_list)

    # def __df__(self) -> pd.DataFrame:
    def to_df(self) -> pd.DataFrame:
        content = []
        for date, text, title, is_update, update_of in zip(self.dates, self.text_list, self.titles, self.is_update, self.update_of):
            content.append({
                "date": date,
                "title": title,
                "text": text,
                "is_update": is_update,
                "update_of": update_of,
            })
        df = pd.DataFrame(content)
        return df

    def __repr__(self) -> str:
        repr_string = self.create_repr_string()
        return repr_string


def safe_instantiate(text):
    try:
        return Report(text)
    except Exception as e:
        print(f"Error instantiating Report with text {text}: {e}")
        return None  # Or return a custom error object


report_list = [safe_instantiate(t) for t in texts]
report_list = [report for report in report_list if report.valid]
# report = Report(foo_str)
# report
report_list[0]

# %%


# print all report titles
for i, report in enumerate(report_list):
    print(i, report.titles[0])

# %%








# %% ####################################################################
#                    Data Processing
# #######################################################################

df = pd.concat([r.to_df() for r in report_list], ignore_index=True)

# Round date to day
df['rounded_date'] = df['date'].dt.floor('d')

df['length'] = df['text'].apply(len)

# Bin the text length and convert to strings
df['length_bin'] = pd.cut(df['length'], bins=list(range(0, 10000, 100)))
df['length_bin'] = df['length_bin'].astype(str)

# Count occurrences in each bin
length_bin_counts = df['length_bin'].value_counts().reset_index()
length_bin_counts.columns = ['text_length_bin', 'count']

# Convert string intervals back to pd.Interval for sorting
length_bin_counts['text_length_bin'] = length_bin_counts['text_length_bin'].apply(lambda x: pd.Interval(left=int(x.split(',')[0][1:]), right=int(x.split(',')[1][:-1]), closed='right'))

# Sort by the interval
length_bin_counts = length_bin_counts.sort_values('text_length_bin')

# Convert intervals back to strings for plotting
length_bin_counts['text_length_bin'] = length_bin_counts['text_length_bin'].astype(str)












# %%

min_date = df['rounded_date'].min()
max_date = df['rounded_date'].max()
# create a list of dates between min and max date
date_range = pd.date_range(start=min_date, end=max_date, freq='D')
date_df = pd.DataFrame(date_range, columns=['rounded_date'])

# merge date_df with df
foo_df = pd.merge(date_df, df, on='rounded_date', how='left')
foo_df['_value'] = 0
# where text is not null, set _value to 100
foo_df.loc[foo_df['text'].notnull(), '_value'] = 100

# apply rolling mean to _value
# foo_df['_value'] = foo_df['_value'].rolling(window=7, center=True).mean()
foo_df['_value'] = foo_df['_value'].rolling(window=30, center=True).mean()
foo_df['_value'] = foo_df['_value'].rolling(window=10, center=True).mean()

foo_df['year'] = foo_df['rounded_date'].dt.year
# create a column equals to 'rounded_date', but change the year to 2000
foo_df['date_2000'] = foo_df['rounded_date'].apply(lambda x: x.replace(year=2000))

# foo_df


# plot _value vs date
# fig = px.line(foo_df, x='date', y="_value")
# fig.show()




# %%

# plot histogram of text length
df['length'].hist(bins=50)



# %% ####################################################################
#                    Plot value counts of is_update
# #######################################################################

def plot_main_vs_update(df):
    fig, axes = plt.subplots(figsize=(6,3))
    df['is_update'].value_counts().plot(kind='bar')
    # add number to the bar
    for i, v in enumerate(df['is_update'].value_counts()):
        plt.text(i, v + 5, str(v), ha='center')

    axes.spines[['right', 'top']].set_visible(False)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=0, ha='right')
    return fig

plot_main_vs_update(df)

# %%
# %% ####################################################################
#                    Text Length Distribution
# #######################################################################

def plotly_text_length_dist(df):

    fig=go.Figure()
    fig.add_trace(
        go.Bar(
            x=df['length_bin'],
            y=np.ones(df['length_bin'].shape[0]),
            hovertext=df['title'],  # Add title as hover text
            hoverinfo='text',  # Display only the hover text

            marker=dict(
                color=df['length'],  # Use text length for color
                colorscale='Viridis',  # Choose a colorscale
                colorbar=dict(title='Text Length')  # Add a colorbar
            ),

            # add red line width if is update
            marker_line_width=df['is_update'].map({True: 2, False: 0}),
            marker_line_color='red',
        )
    )

    fig.update_layout(
        barmode="stack",
        title='Text Length Distribution',
        title_x=0.5,
        xaxis=dict(
            categoryorder='array',
            categoryarray=length_bin_counts['text_length_bin'].tolist(),
            # categoryarray=df['length_bin'].tolist(),
        ),
        # yaxis=dict(visible=False),
    )
    return fig

plotly_text_length_dist(df)


# %% ####################################################################
#                    Gitlike-Heatmap-Calplot
# #######################################################################

years = [2015, 2020, 2021]

from plotly_calplot import calplot
fig = calplot(
    data=foo_df.loc[foo_df['rounded_date'].dt.year.isin(years)],
    x="rounded_date",
    y="_value",
    gap=1,
    # dark_theme=True,
    text="title",
    title="Activity Chart",
    # space_between_plots=0.08,
)
fig.update_layout(title_x=0.5)
fig.show()




# %% ####################################################################
#                    WIP
# #######################################################################


years = [2020, 2021] # foo_df['date'].dt.year.unique()
# fig = make_subplots(rows=len(years), cols=1, subplot_titles=years)

# def display_year(data, year: int = None, fig=None, row: int = None):
def display_year(data, year: int, fig=None, row: int = None):
    # if year is None:
    #     year = datetime.datetime.now().year

    d1 = datetime.date(year, 1, 1)
    d2 = datetime.date(year, 12, 31)
    delta = d2 - d1

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_days =   [31,    28,    31,     30,    31,     30,    31,    31,    30,    31,    30,    31]
    # create month_starts as datetimes
    month_starts = [datetime.date(year, i+1, 1) for i in range(12)]
    # month_starts = [0] + list(np.cumsum(month_days)[:-1])
    # month_ends = np.cumsum(month_days)
    # month_positions = (np.cumsum(month_days) - 15)/7
    month_positions = (np.cumsum(month_days) - 15)
    # print(month_positions)
    dates_in_year = [d1 + datetime.timedelta(i) for i in range(delta.days+1)] # gives me a list with datetimes for each day a year
    weekdays_in_year = [i.weekday() for i in dates_in_year] # gives [0,1,2,3,4,5,6,0,1,2,3,4,5,6,…] (ticktext in xaxis dict translates this to weekdays

    weeknumber_of_dates = [int(i.strftime("%V")) if not (int(i.strftime("%V")) == 1 and i.month == 12) else 53
                           for i in dates_in_year] # gives [1,1,1,1,1,1,1,2,2,2,2,2,2,2,…] name is self-explanatory
    text = [str(i) for i in dates_in_year] # gives something like list of strings like ‘2018-01-25’ for each date. Used in data trace to make good hovertext.
    #4cc417 green #347c17 dark green
    colorscale=[[False, '#eeeeee'], [True, '#76cf63']]

    # handle end of year

    y = foo_df.loc[foo_df['date'].dt.year == year, '_value']
    y_max = y.max()

    # plot line _value
    data = [
        go.Scatter(
            x=foo_df.loc[foo_df['date'].dt.year == year, 'date'],
            y=y,
            # y=foo_df['_value'],
            mode='lines',
            line=dict(
                color='blue',
                width=1,
            ),
            hoverinfo='text',
        )
    ]

    # plot a green vertical line for dates where the column text is not null
    data += [
        go.Scatter(
            x=foo_df.loc[(foo_df['date'].dt.year == year) & (foo_df['text'].notnull()), 'date'],
            y=[0]*foo_df.loc[(foo_df['date'].dt.year == year) & (foo_df['text'].notnull()), 'date'].shape[0],
            mode='markers',
            marker=dict(
                color='green',
                size=10,
                symbol='arrow',
            ),
            # add title as hover text
            text=foo_df.loc[(foo_df['date'].dt.year == year) & (foo_df['text'].notnull()), 'title'],
            hoverinfo='text',
            # hoverinfo='skip'
        )
    ]
    

    layout = go.Layout(
        # title='activity chart',
        height=250,
        font={'size':10, 'color':'#9e9e9e'},
        plot_bgcolor=('#fff'),
        margin = dict(t=40),
        showlegend=False
    )

    # show month_starts as vertical lines
    for month_start in month_starts:
        data += [
            go.Scatter(
                x=[month_start, month_start],
                y=[0, y_max],
                mode='lines',
                line=dict(
                    color='red',
                    width=1
                ),
                hoverinfo='skip'
            )
        ]


    if fig is None:
        # make y shared
        fig = go.Figure(data=data, layout=layout)
    else:
        fig.add_traces(data, rows=[(row+1)]*len(data), cols=[1]*len(data))
        fig.update_layout(layout)
        fig.update_xaxes(layout['xaxis'])
        fig.update_yaxes(layout['yaxis'])

    return fig


# def display_years(data, z, years):
def display_years(data, years):
    fig = make_subplots(rows=len(years), cols=1, subplot_titles=years, shared_yaxes='all', vertical_spacing=0.1)
    for i, year in enumerate(years):
        # data = z[i*365 : (i+1)*365]
        display_year(data, year=year, fig=fig, row=i)
        fig.update_layout(height=250*len(years))
        
    return fig


# display_year(foo_df, year=2017, fig=None, row=None)
display_years(foo_df, years=[2017, 2018, 2019, 2020, 2021])




# %% ####################################################################
#                    Aesthetic timeline
# #######################################################################

def aesthetic_timeline(df):

    # plot lineplot of _value vs date for 2020 using seaborn
    years = sorted(df['year'].unique())
    pal = sns.cubehelix_palette(len(years), rot=-.25, light=.7)
    month_middle = [datetime.date(2000, i+1, 15) for i in range(12)]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig, axes = plt.subplots(nrows=len(years), figsize=(10, 10), sharex=True, sharey=True)

    for i, year in enumerate(years):
        sns.lineplot(data=df.loc[df['year'] == year], x='date_2000', y='_value', ax=axes[i], color=pal[i])
        axes[i].fill_between(df.loc[df['year'] == year]['date_2000'], df.loc[df['year'] == year]['_value'], alpha=1, color=pal[i])
        sns.lineplot(data=df.loc[df['year'] == year], x='date_2000', y='_value', ax=axes[i], color='white', linewidth=2)
        axes[i].set_xlabel('')
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['bottom'].set_color(pal[i])
        axes[i].text(-0.01, .2, year, fontweight="bold", color=pal[i], ha="left", va="center", transform=axes[i].transAxes)
        axes[i].set_yticks([])
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')
        # set xticks to month names
        axes[i].set_xticks(month_middle)
        # axes[i].set_xticklabels(month_names)
        # use the month names as xticks
        axes[i].set_xticklabels(month_names)

    plt.subplots_adjust(hspace=-.25)
    return fig

aesthetic_timeline(foo_df)


# %%


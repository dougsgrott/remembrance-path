# %%

import json
import codecs
import copy
import re
import textwrap
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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


foo_str = """|Iniciado: 19770101 16h48 Aubade
This is a special way of being afraid
No trick dispels. Religion used to try,
That vast moth-eaten musical brocade
Created to pretend we never die,
And specious stuff that says No rational being
Can fear a thing it will not feel, not seeing
That this is what we fear—no sight, no sound,
No touch or taste or smell, nothing to think with,
Nothing to love or link with,
The anaesthetic from which none come round.

|Atualizado: 20240512 23h53
And so it stays just on the edge of vision,
A small unfocused blur, a standing chill
That slows each impulse down to indecision. 

|Atualizado: 20240513 00h16
Most things may never happen: this one will.

|========================================================="""

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
        self.dates = [datetime.strptime(d, '%Y%m%d %Hh%M') for d in self.string_dates]

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


df = pd.concat([r.to_df() for r in report_list], ignore_index=True)
df


# %% ####################################################################
#                    Plot value counts of is_update
# #######################################################################

fig, axes = plt.subplots(figsize=(6,3))
df['is_update'].value_counts().plot(kind='bar')
# add number to the bar
for i, v in enumerate(df['is_update'].value_counts()):
    plt.text(i, v + 5, str(v), ha='center')

axes.spines[['right', 'top']].set_visible(False)
axes.set_xticklabels(axes.get_xticklabels(), rotation=0, ha='right')

# %%




# %% ####################################################################
#                    Data Processing
# #######################################################################

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

# plot histogram of text length
df['length'].hist(bins=50)

# %% ####################################################################
#                    Text Length Distribution
# #######################################################################

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

# %%

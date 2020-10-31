# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import schemdraw
from schemdraw import flow

# +
#Long strings for below

verification_exclusions = 'Newly registered\n(n=4366)\nFully completed\nw/ results\n(n=2914)\nPending trials\n(n=593)'

qc_results = 'Pending QC\n(n=601)\nCompleted QC\n(n=3221)\nTotal Submitted\n(n=3821)'

# +
d = schemdraw.Drawing()
#Getting to Covered Trials
total = d.add(flow.Box(w=10, h=2, label= 'Registered Trials on ClinicalTrials.gov\n(N=327154)'))
d.add(flow.Arrow('down'))
d.add(flow.Arrow('right', at=(0, -3.5)))
not_covered = d.add(flow.Box(w=5, h=2, label='Not identifiable\nas FDAAA covered\n(n=304252)', anchor='W'))
pop = d.add(flow.Box(w=30, h=1, at=(0,-5), label='All Covered Trials (N=22902)'))

#Registration Pop
d.add(flow.Line('down',  at=(-12,-6), l=1))
reg = d.add(flow.Box(w=4, h=2, label='Registration\n(No exclusions)'))
d.add(flow.Arrow('down', l=2))
pop_reg = d.add(flow.Box(w=4.5, h=2, label='Registered Trials\n(n=22902)'))

#Verification Pop
d.add(flow.Line('down', at=(-7, -6), l=1))
veri = d.add(flow.Box(w=3, h=1, label = 'Verification'))
d.add(flow.Line('down'))
d.add(flow.Arrow('right', l=1))
veri_exclude = d.add(flow.Box(w=3, h=3, label = verification_exclusions, fontsize=9, anchor='W'))
d.add(flow.Arrow('down', at=(-7,-11), l=3))
veri_pop = d.add(flow.Box(w=4.5, h=2, label='Require Verification\n(n=15029)'))

#Certificate of Delay Pop
d.add(flow.Line('down', at=(0, -6), l=1))
cod = d.add(flow.Box(w=3, h=1.5, label = 'Certificates\nof Delay'))
d.add(flow.Line('down', l=2))
d.add(flow.Arrow('right', l=1))
cod_exclude = d.add(flow.Box(w=3, h=1.5, label=f'No Certificate\n(n={22902-857})', anchor='W', fontsize=10))
d.add(flow.Arrow('down', at=(0, -10.5), l=2.5))
certificate_pop = d.add(flow.Box(w=4.5, h=2, label='Has Certificate\n(n=857)'))

#QC Pop
d.add(flow.Line('down', at=(6, -6), l=1))
qc = d.add(flow.Box(w=3, h=1, label = 'QC Analysis'))
d.add(flow.Line('down', l=1.5))
d.add(flow.Arrow('right', l=1))
no_results = d.add(flow.Box(w=3, h=2, label=f'No Results\nSubmission\n(n={22902-3822})', anchor='W', fontsize=10))
d.add(flow.Arrow('down', at=(6, -9.5), l=1.5))
results = d.add(flow.Box(w=3, h=3, label=qc_results, fontsize=10))
d.add(flow.Line('down', l=1.5))
d.add(flow.Arrow('right', l=1))
no_detail = d.add(flow.Box(w=2.5, h=1.5, label=f'No QC Detail\n(n={3822-3793})', anchor='W', fontsize=10))
d.add(flow.Arrow('down', at=(6, -15.5), l=1.5))
detailed = d.add(flow.Box(w=3, h=2, label="Detailed QC\nData\n(n=3793)"))

#Results Docs Pop
d.add(flow.Line('down', at=(12,-6), l=1))
docs = d.add(flow.Box(w=3, h=1.5, label='Results\nDocuments'))
d.add(flow.Line('down', l=1.5))
d.add(flow.Arrow('right', l=1))
not_due = d.add(flow.Box(w=3, h=1.5, label=f'Results Not Due\n(n={22902-5320})', anchor='W', fontsize=10))
d.add(flow.Arrow('down', at=(12, -10), l=2))
due = d.add(flow.Box(w=3, h=1.5, label='Results Due\n(n=5320)'))
d.add(flow.Line('down', l=1.5))
d.add(flow.Arrow('right', l=1))
no_results = d.add(flow.Box(w=3, h=2, label='No Results\n(n=1911)\nPending Results\n(n=476)', anchor='W', fontsize=10))
d.add(flow.Arrow('down', l=2, at=(12, -15)))
due_results = d.add(flow.Box(w=3.5, h=2, label="Due w/ Results\n(n=2933)"))
#d.save('figures/flowchart.svg')
d.draw()
# -



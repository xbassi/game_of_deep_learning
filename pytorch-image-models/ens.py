import pandas as pd
import os


p1 = pd.read_csv('./pred/p1.csv')
p2 = pd.read_csv('./pred/p2.csv')
p3 = pd.read_csv('./pred/p3.csv')
p4 = pd.read_csv('./pred/p4.csv')
p10 = pd.read_csv('./pred/p10.csv')


p = pd.DataFrame()

p['1'] = ( p1['l1'] + p2['l1'] + p3['l1'] + p4['l1'] + p10['l1'])#+ p8['l1'])  
p['2'] = ( p1['l2'] + p2['l2'] + p3['l2'] + p4['l2'] + p10['l2'])#+ p8['l2'])  
p['3'] = ( p1['l3'] + p2['l3'] + p3['l3'] + p4['l3'] + p10['l3'])#+ p8['l3'])  
p['4'] = ( p1['l4'] + p2['l4'] + p3['l4'] + p4['l4'] + p10['l4'])#+ p8['l4'])  
p['5'] = ( p1['l5'] + p2['l5'] + p3['l5'] + p4['l5'] + p10['l5'])#+ p8['l5'])  


k = pd.DataFrame()

k['image'] = p2['image']
k['category'] = p.idxmax(axis=1, skipna=True)


# print(k)


k.to_csv('123410.csv',index=False)







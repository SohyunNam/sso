import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # ### data pre-processing
activity_data = pd.read_excel('../data/Layout_Activity_A0001.xlsx')
bom_data = pd.read_excel('../data/Layout_BOM_A0001.xlsx')
#
# # # relation
# # relation_data = pd.DataFrame(columns=['pre', 'post', 'relation'])
# # relation_data['pre'] = data['선행']
# # relation_data['post'] = data['후행']
# # relation_data['relation'] = data['관계']
# #
# activity
activity_data['시작일'] = pd.to_datetime(activity_data['시작일'], format='%Y%m%d')
activity_data['종료일'] = pd.to_datetime(activity_data['종료일'], format='%Y%m%d')
initial_date = activity_data['시작일'].min()

activity_data['시작일'] = activity_data['시작일'].apply(lambda x: (x - initial_date).days)
activity_data['종료일'] = activity_data['종료일'].apply(lambda x: (x - initial_date).days)
#
# '''
# define variable such as process time, block code..etc. for simulating
# '''
# # activity_data['process time'] = list(activity_data['종료일'] - activity_data['시작일'] + 1)
# activity_data['호선'] = activity_data['호선'].apply(lambda x: str(x))
# activity_data['블록'] = activity_data['블록'].apply(lambda x: str(x) if not x == 322 else '322E0')
# activity_data['block code'] = activity_data['호선'] + '_' + activity_data['블록']
#
# # description_list = []
# # for i in range(len(activity_data)):
# #     temp = activity_data.iloc[i]
# #     description_list.append(temp['ACT설명'][len(temp['블록']) + 1:])
# # activity_data['data description'] = description_list
# bom_data['호선'] = bom_data['호선'].apply(lambda x: str(x))
# bom_data['블록'] = bom_data['블록'].apply(lambda x: str(x))
# bom_data['상위블록'] = bom_data['상위블록'].apply(lambda x: str(x))
# bom_data['block code'] = bom_data['호선'] + '_' + bom_data['블록']
# #
# # x = list(bom_data['block code'])
# # length = list(bom_data['길이'])
# # width = list(bom_data['폭'])
# # plt.plot(x, length)
# # plt.xlabel('block')
# # plt.ylabel('length')
# # plt.show()
# #
# # plt.plot(x, width)
# # plt.xlabel('block')
# # plt.ylabel('width')
# # plt.show()
# # bom_data.to_excel('../data/block_size.xlsx')
# #
# #
# # print(0)
# #
# #
# data_size = pd.read_excel('../data/EPL 데이터 정리.xlsx')
# act_before_prePE = activity_data[(activity_data['공정공종'] == 'C11') | (activity_data['공정공종'] == 'C12') | \
#                                   (activity_data['공정공종'] == 'C13') | (activity_data['공정공종'] == 'C14') | \
#                                   (activity_data['공정공종'] == 'C15') | (activity_data['공정공종'] == 'C21') | \
#                                   (activity_data['공정공종'] == 'C91')]
# list_before_prePE = list(act_before_prePE.drop_duplicates(['block code'])['block code'])
# bom_before_prePE = bom_data[bom_data['block code'].isin(list_before_prePE)]
# bom_before_prePE.to_excel('../data/before_prePE.xlsx')
#
# act_MUNIT = activity_data[(activity_data['공정공종'] == 'F22')]
# list_MUNIT = list(act_MUNIT.drop_duplicates(['block code'])['block code'])
# bom_MUNIT = bom_data[bom_data['block code'].isin(list_MUNIT)]
# bom_MUNIT.to_excel('../data/MUNIT.xlsx')
#
# act_prePE = activity_data[(activity_data['공정공종'] == 'F21') | (activity_data['공정공종'] == 'F61') | \
#                           (activity_data['공정공종'] == 'G41') | (activity_data['공정공종'] == 'G4A') | \
#                           (activity_data['공정공종'] == 'G4B') | (activity_data['공정공종'] == 'G61') | \
#                           (activity_data['공정공종'] == 'GX3') | (activity_data['공정공종'] == 'H32') | \
#                           (activity_data['공정공종'] == 'H64') | (activity_data['공정공종'] == 'JX1')]
# list_prePE = list(act_prePE.drop_duplicates(['block code'])['block code'])
# bom_prePE = bom_data[bom_data['block code'].isin(list_prePE)]
# bom_prePE.to_excel('../data/prePE.xlsx')

## new pre-processing
activity_data['호선'] = activity_data['호선'].apply(lambda x: str(x))
activity_data['블록'] = activity_data['블록'].apply(lambda x: str(x) if not x == 322 else '322E0')
activity_data['block code'] = activity_data['호선'] + '_' + activity_data['블록']

block_list = list(activity_data.drop_duplicates(['block code'])['block code'])

for block_code in block_list"" \
        ""
print(0)
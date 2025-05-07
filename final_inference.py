import pandas as pd
import numpy as np
import json
import pandas as pd
from itertools import combinations
import numpy as np
import os
import pickle
from xgboost import XGBRegressor
import keras
from keras.models import load_model
import requests
from pulp import *
import datetime
from bs4 import BeautifulSoup
import re
from keras.layers import Layer, Dense, BatchNormalization, Activation, Add, Lambda,Input
import keras.backend as K
name_dict_df=pd.read_csv('data/SquadData_AllTeams-Table 1.csv')
name_dict={}
for i in range(len(name_dict_df)):
 name_dict[name_dict_df['Player Name'][i]]=name_dict_df['player_name'][i]
team_dict={'RCB':"Royal Challengers Bengaluru",'KKR':"Kolkata Knight Riders",'CHE':"Chennai Super Kings",'SRH':"Sunrisers Hyderabad",'GT':"Gujarat Titans",'DC':"Delhi Capitals",'MI':"Mumbai Indians",'LSG':"Lucknow Super Giants",'PBKS':"Punjab Kings",'RR':"Rajasthan Royals"}
with open('data/output.txt', 'r') as convert_file3: 
                dict1=(json.loads(convert_file3.read()))
dict1={key:{eval(key2):val for key2,val in dict1[key].items()} for key in dict1.keys()}
@keras.saving.register_keras_serializable()
class BottleneckResidualBlock(Layer):
    def __init__(self, units=300, **kwargs):
        super(BottleneckResidualBlock, self).__init__().__init__(**kwargs)
        self.units = units
        self.dense1 = Dense(units)
        self.bn1 = BatchNormalization()
        self.activation1 = Activation('relu')
        self.dense2 = Dense(units)
        self.bn2 = BatchNormalization()
        self.activation2 = Activation('relu')
        self.dense3 = Dense(units)
        self.bn3 = BatchNormalization()
        self.activation3 = Activation('relu')
        self.dense4 = Dense(units)
        self.bn4 = BatchNormalization()
        self.activation4 = Activation('relu')

    def call(self, inputs):
        X_shortcut = inputs
        X = self.dense1(inputs)
        X = self.bn1(X)
        X = self.activation1(X)
        X = self.dense2(X)
        X = self.bn2(X)
        X = self.activation2(X)
        X = self.dense3(X)
        X = self.bn3(X)
        X = self.activation3(X)
        X = self.dense4(X)
        X = self.bn4(X)
        X = self.activation4(X)
        X = Add()([X, X_shortcut])
        return self.activation4(X)
    def get_config(self):
        config = super(BottleneckResidualBlock, self).get_config()
        return config
    def build(self, input_shape):
        # The build method is called with input_shape the first time the layer is used
        # No need to define anything here unless your internal layers need dynamic input
        pass  # Let internal Keras layers handle their own weight initialization
def pop_inf(file,venue,toss_bat,date):

            with open('data/venue_dict.txt', 'r') as convert_file: 
               venue_dict=(json.loads(convert_file.read()))  
            with open('data/venue_dict_bowl.txt', 'r') as convert_file5: 
               venue_dict_bowl=(json.loads(convert_file5.read()))
            with open('data/venue_dict_field.txt', 'r') as convert_file6: 
               venue_dict_field=(json.loads(convert_file6.read()))      
            with open('data/venue_player_dict.txt', 'r') as convert_file2: 
                venue_player_dict=(json.loads(convert_file2.read()))
            with open('data/opposition_player_dict.txt', 'r') as convert_file3: 
                opposition_player_dict=(json.loads(convert_file3.read()))
            with open('data/opposition_dict.txt', 'r') as convert_file4: 
                opposition_dict=(json.loads(convert_file4.read()))
            inference_batsman_details=pd.DataFrame() 
            inference_bowler_details=pd.DataFrame()        
            inference_bowler_details_venue=pd.DataFrame() 
            player_fielding=pd.DataFrame()
            venue_player_df=pd.DataFrame(columns = ['date', 'name', 'opposition_name', 'venue_name', 'team', 'innings', 'vbaHr_4s_agg', 'vbaHr_6s_agg', 's4/inning', 's6/inning', 'p6/inning', 'p4/inning', 'sw/pw', 'spe/pe', 'caught', 'runouts', 'stumpings', 'bowled', 'total_wicks', 'dismissals/innings', 'dismissals_b1/innings', 'dismissals_b2/innings', 'dismissals_b3/innings', 'bowled_b1/innings', 'bowled_b2/innings', 'bowled_b3/innings', 'madien_b1/innings', 'madien_b2/innings', 'madien_b3/innings', 'dismissals_b1', 'dismissals_b2', 'dismissals_b3', 'bowled_b1', 'bowled_b2', 'bowled_b3', 'madien_b1', 'madien_b2', 'madien_b3', 'runouts/i', 'caught/i', 'stumped/i', 'bowled/i', 'spin_runs_b1', 'spin_runs_b2', 'spin_runs_b3', 'vbaHs_4s_1', 'vbaHs_4s_2', 'vbaHs_4s_3', 'vbaHs_6s_1', 'vbaHs_6s_2', 'vbaHs_6s_3', 'vbaHs_dismissals_1', 'vbaHs_dismissals_2', 'vbaHs_dismissals_3', 'spin_balls_b1', 'spin_balls_b2', 'spin_balls_b3', 'pace_runs_b1', 'pace_runs_b2', 'pace_runs_b3', 'vbaHp_dismissals_1', 'vbaHp_dismissals_2', 'vbaHp_dismissals_3', 'pace_balls_b1', 'pace_balls_b2', 'pace_balls_b3', 'vbaHp_6s_1', 'vbaHp_6s_2', 'vbaHp_6s_3', 'vbaHp_4s_1', 'vbaHp_4s_2', 'vbaHp_4s_3', 'vbaHs_economy_1', 'vbaHp_economy_1', 'vbaHs_economy_2', 'vbaHp_economy_2', 'vbaHs_economy_3', 'vbaHp_economy_3', 'vbaHr_economy_1', 'vbaHr_economy_2', 'vbaHr_economy_3', 'vbaHr_4s_1', 'vbaHr_4s_2', 'vbaHr_4s_3', 'vbaHr_6s_1', 'vbaHr_6s_2', 'vbaHr_6s_3', 'vbaHr_dismissals_1', 'vbaHr_dismissals_2', 'vbaHr_dismissals_3', 'vbaHs_economy_agg', 'vbaHp_economy_agg', 'vbaHr_economy_agg', 'vbaHs_4s_agg', 'vbaHp_4s_agg', 'vbaHs_dismissals_agg', 'vbaHp_dismissals_agg', 'vbaHr_dismissals_agg', 'vbaHs_6s_agg', 'vbaHp_6s_agg', 'average', 'average_4s', 'average_6s', 'average_balls_involved', 'balls_involved', 'balls']
)
            overall_batsman_details=pd.read_pickle('data/pickles/overall_batsman_details.pkl').fillna(0)
            overall_bowler_details=pd.read_pickle('data/pickles/overall_bowler_details.pkl').fillna(0)
            overall_fielding_details=pd.read_pickle('data/pickles/overall_fielding_details.pkl').fillna(0)
            # overall_batsman_details.index=overall_batsman_details['player_name']
            # overall_bowler_details.index=overall_bowler_details['player_name']
            # overall_fielding_details.index=overall_fielding_details['player_name']
            match_bowler_details=pd.read_csv('data/match_bowler_details.csv')
            # match_details, player_details, wickets, wickets_field= extract_details(fil_loc)
            try:
             df=pd.read_csv(file)
            except:
                df=pd.read_excel(file)
            player_grp=df.groupby('IsPlaying')
            playing_df=player_grp.get_group('PLAYING')
            playing_df=playing_df.reset_index()
            for i in range(len(playing_df)):
                if playing_df.loc[i,'Team']==toss_bat:
                    playing_df.loc[i,'inning_no']=0
                else:
                    playing_df.loc[i,'inning_no']=1 
            playing_df['Team']=[team_dict[i] for i in playing_df['Team']]
            teams=playing_df['Team'].unique()

            playing_df['player_name']=[name_dict[player_name] if player_name in name_dict.keys() and pd.isna(name_dict[player_name])==False else player_name for player_name in playing_df['Player Name']]

            print(playing_df)
            
            # print(playing_df['Opposition'])
            for i in range(len(playing_df)):
                if playing_df.loc[i,'Team']==teams[0]:
                    playing_df.loc[i,'Opposition']=teams[1]
                else:
                    playing_df.loc[i,'Opposition']=teams[0]    
            

            venue=venue
            # overall_batsman_details=overall_batsman_details.astype(object)
            for player_name in playing_df['player_name']:
             
                player=player_name
                print(playing_df['Opposition'])
                print(player_name)
                opposition = playing_df['Opposition'][playing_df['player_name']==player_name].values[0] 
                if venue not in venue_dict: 
                     
                    venue_dict[venue]={}
                    venue_dict[venue]['date'] = ''
                    venue_dict[venue]['name'] = ''
                    venue_dict[venue]['opposition_name'] = '' 

                    venue_dict[venue]['venue_name'] = ''
                    venue_dict[venue]['team'] = ''
                    venue_dict[venue]['innings'] = 0
                    venue_dict[venue]['s4/p4'] = 0
                    venue_dict[venue]['s6/p6'] = 0
                    venue_dict[venue]['s4/inning'] = 0
                    venue_dict[venue]['s6/inning'] = 0
                    venue_dict[venue]['p6/inning'] = 0
                    venue_dict[venue]['p4/inning'] = 0
                    venue_dict[venue]['sw/pw']=0
                    venue_dict[venue]['spe/pe'] =0
                    venue_dict[venue]['caught'] =0
                    venue_dict[venue]['runouts'] =0
                    venue_dict[venue]['stumpings'] =0
                    venue_dict[venue]['bowled'] =0
                    venue_dict[venue]['total_wicks'] =0
                    venue_dict[venue]['dismissals/innings']=0
                    venue_dict[venue]['dismissals_b1/innings']=0
                    venue_dict[venue]['dismissals_b2/innings']=0
                    venue_dict[venue]['dismissals_b3/innings']=0
                    venue_dict[venue]['bowled_b1/innings']=0
                    venue_dict[venue]['bowled_b2/innings']=0
                    venue_dict[venue]['bowled_b3/innings']=0
                    venue_dict[venue]['madien_b1/innings']=0
                    venue_dict[venue]['madien_b2/innings']=0
                    venue_dict[venue]['madien_b3/innings']=0
                    venue_dict[venue]['dismissals/innings']=0
                    venue_dict[venue]['dismissals_b1']=0
                    venue_dict[venue]['dismissals_b2']=0
                    venue_dict[venue]['dismissals_b3']=0
                    venue_dict[venue]['bowled_b1']=0
                    venue_dict[venue]['bowled_b2']=0
                    venue_dict[venue]['bowled_b3']=0
                    venue_dict[venue]['madien_b1']=0
                    venue_dict[venue]['madien_b2']=0
                    venue_dict[venue]['madien_b3']=0
                    venue_dict[venue]['runouts/i']=0
                    venue_dict[venue]['caught/i']=0
                    venue_dict[venue]['stumped/i']=0
                    venue_dict[venue]['bowled/i']=0
                    venue_dict[venue]['spin_runs_b1']=0
                    venue_dict[venue]['spin_runs_b2']=0
                    venue_dict[venue]['spin_runs_b3']=0
                    venue_dict[venue]['vHs_4s_1']=0
                    venue_dict[venue]['vHs_4s_2']=0
                    venue_dict[venue]['vHs_4s_3']=0
                    venue_dict[venue]['vHs_6s_1']=0
                    venue_dict[venue]['vHs_6s_2']=0
                    venue_dict[venue]['vHs_6s_3']=0
                    venue_dict[venue]['vHs_dismissals_1']=0
                    venue_dict[venue]['vHs_dismissals_2']=0
                    venue_dict[venue]['vHs_dismissals_3']=0
                    venue_dict[venue]['spin_balls_b1']=0
                    venue_dict[venue]['spin_balls_b2']=0
                    venue_dict[venue]['spin_balls_b3']=0
                    venue_dict[venue]['pace_runs_b1']=0
                    venue_dict[venue]['pace_runs_b2']=0
                    venue_dict[venue]['pace_runs_b3']=0
                    venue_dict[venue]['vHp_dismissals_1']=0
                    venue_dict[venue]['vHp_dismissals_2']=0
                    venue_dict[venue]['vHp_dismissals_3']=0
                    venue_dict[venue]['pace_balls_b1']=0
                    venue_dict[venue]['pace_balls_b2']=0
                    venue_dict[venue]['pace_balls_b3']=0
                    venue_dict[venue]['vHp_6s_1']=0
                    venue_dict[venue]['vHp_6s_2']=0
                    venue_dict[venue]['vHp_6s_3']=0
                    venue_dict[venue]['vHp_4s_1']=0
                    venue_dict[venue]['vHp_4s_2']=0
                    venue_dict[venue]['vHp_4s_3']=0 
                try:
                    overall_batsman_details.index.get_loc(player)
                except:
                    
                    lst=[0]
                    overall_batsman_details.loc[player,'team'] =playing_df['Team'][playing_df['player_name']==player_name].values[0] 
                    overall_batsman_details.loc[player,'innings'] =list(list([0]))
                    #print(overall_batsman_details)
                    overall_batsman_details.loc[player,'runs'] =list(list([0]))
                    overall_batsman_details.loc[player,'balls'] =list(list([0]))
                    overall_batsman_details.loc[player,'balls_involved'] =list(list([0]))
                    overall_batsman_details.loc[player,'outs'] =list(list([0]))
                    overall_batsman_details.loc[player,'strike_rate'] =list(list([0]))
                    overall_batsman_details.loc[player,'average'] =list(list([0]))
                    overall_batsman_details.loc[player,'zeros'] =list(list([0]))
                    overall_batsman_details.loc[player,'thirties'] =list(list([0]))
                    overall_batsman_details.loc[player,'fifties'] =list(list([0]))
                    overall_batsman_details.loc[player,'centuries'] =list(list([0]))
                    overall_batsman_details.loc[player,'4s'] =list(list([0]))
                    overall_batsman_details.loc[player,'6s'] =list(list([0]))
                    
                    overall_batsman_details.loc[player,'spin_runs_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_runs_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_runs_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_runs_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_runs_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_runs_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_balls_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_balls_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_balls_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_balls_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_balls_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_balls_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_wickets_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_wickets_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_wickets_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_wickets_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_wickets_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_wickets_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_4s_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_4s_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_4s_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_4s_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_4s_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_4s_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_6s_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_6s_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'spin_6s_b3']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_6s_b1']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_6s_b2']=list(list([0]))
                    overall_batsman_details.loc[player,'pace_6s_b3']=list(list([0]))
                if player not in venue_dict:
                        venue_dict[player] = {}
                if venue not in venue_dict[player]:
                        venue_dict[player][venue] = {}
                        venue_dict[player][venue]['innings'] = 0
                        venue_dict[player][venue]['runs'] = 0
                        venue_dict[player][venue]['balls'] = 0
                        venue_dict[player][venue]['wickets'] = 0
                        venue_dict[player][venue]['strike_rate'] = 0
                        venue_dict[player][venue]['average'] = 0
                        venue_dict[player][venue]['previous_average'] = 0
                        venue_dict[player][venue]['centuries'] = 0
                        venue_dict[player][venue]['fifties'] = 0
                        venue_dict[player][venue]['thirties'] = 0
                        venue_dict[player][venue]['zeros'] = 0
                        venue_dict[player][venue]['high_score'] = 0 
                    
                if player not in opposition_dict:
                      opposition_dict[player] = {}      
                if opposition not in opposition_dict[player]:
                        opposition_dict[player][opposition] = {}
                        opposition_dict[player][opposition]['innings'] = 0
                        opposition_dict[player][opposition]['runs'] = 0
                        opposition_dict[player][opposition]['balls'] = 0
                        opposition_dict[player][opposition]['wickets'] = 0
                        opposition_dict[player][opposition]['strike_rate'] = 0
                        opposition_dict[player][opposition]['average'] = 0
                        opposition_dict[player][opposition]['centuries'] = 0
                        opposition_dict[player][opposition]['fifties'] = 0
                        opposition_dict[player][opposition]['thirties'] = 0
                        opposition_dict[player][opposition]['zeros'] = 0
                        opposition_dict[player][opposition]['high_score'] = 0        
                opposition_dict[player][opposition]['innings'] = int(opposition_dict[player][opposition]['innings'])
                opposition_dict[player][opposition]['runs'] = int(opposition_dict[player][opposition]['runs'])
                opposition_dict[player][opposition]['balls'] = int(opposition_dict[player][opposition]['balls'])
                opposition_dict[player][opposition]['wickets'] = int(opposition_dict[player][opposition]['wickets'])
                opposition_dict[player][opposition]['strike_rate'] = int(opposition_dict[player][opposition]['strike_rate'])
                opposition_dict[player][opposition]['average'] = int(opposition_dict[player][opposition]['average'])
                opposition_dict[player][opposition]['centuries'] = int(opposition_dict[player][opposition]['centuries'])
                opposition_dict[player][opposition]['fifties'] = int(opposition_dict[player][opposition]['fifties'])
                opposition_dict[player][opposition]['thirties'] = int(opposition_dict[player][opposition]['thirties'])
                opposition_dict[player][opposition]['zeros'] = int(opposition_dict[player][opposition]['zeros'])
                opposition_dict[player][opposition]['high_score'] = int(opposition_dict[player][opposition]['high_score'])     
                if venue not in venue_player_dict.keys():
                    venue_player_dict[venue]={}
                if player not in venue_player_dict[venue].keys():
                    venue_player_dict[venue][player]={}
                    venue_player_dict[venue][player]['date'] = ''
                    venue_player_dict[venue][player]['name'] = ''
                    venue_player_dict[venue][player]['opposition_name'] = '' 

                    venue_player_dict[venue][player]['venue_name'] = ''
                    venue_player_dict[venue][player]['team'] = ''
                    venue_player_dict[venue][player]['innings'] = 0
                    venue_player_dict[venue][player]['vbaHr_4s_agg'] = 0
                    venue_player_dict[venue][player]['vbaHr_6s_agg'] = 0
                    venue_player_dict[venue][player]['s4/inning'] = 0
                    venue_player_dict[venue][player]['s6/inning'] = 0
                    venue_player_dict[venue][player]['p6/inning'] = 0
                    venue_player_dict[venue][player]['p4/inning'] = 0
                    venue_player_dict[venue][player]['sw/pw']=0
                    venue_player_dict[venue][player]['spe/pe'] =0
                    venue_player_dict[venue][player]['caught'] =0
                    venue_player_dict[venue][player]['average'] =0
                    venue_player_dict[venue][player]['runouts'] =0
                    venue_player_dict[venue][player]['stumpings'] =0
                    venue_player_dict[venue][player]['bowled'] =0
                    venue_player_dict[venue][player]['total_wicks'] =0
                    venue_player_dict[venue][player]['dismissals/innings']=0
                    venue_player_dict[venue][player]['dismissals_b1/innings']=0
                    venue_player_dict[venue][player]['dismissals_b2/innings']=0
                    venue_player_dict[venue][player]['dismissals_b3/innings']=0
                    venue_player_dict[venue][player]['bowled_b1/innings']=0
                    venue_player_dict[venue][player]['bowled_b2/innings']=0
                    venue_player_dict[venue][player]['bowled_b3/innings']=0
                    venue_player_dict[venue][player]['madien_b1/innings']=0
                    venue_player_dict[venue][player]['madien_b2/innings']=0
                    venue_player_dict[venue][player]['madien_b3/innings']=0
                    venue_player_dict[venue][player]['dismissals/innings']=0
                    venue_player_dict[venue][player]['dismissals_b1']=0
                    venue_player_dict[venue][player]['dismissals_b2']=0
                    venue_player_dict[venue][player]['dismissals_b3']=0
                    venue_player_dict[venue][player]['bowled_b1']=0
                    venue_player_dict[venue][player]['bowled_b2']=0
                    venue_player_dict[venue][player]['bowled_b3']=0
                    venue_player_dict[venue][player]['madien_b1']=0
                    venue_player_dict[venue][player]['madien_b2']=0
                    venue_player_dict[venue][player]['madien_b3']=0
                    venue_player_dict[venue][player]['runouts/i']=0
                    venue_player_dict[venue][player]['caught/i']=0
                    venue_player_dict[venue][player]['stumped/i']=0
                    venue_player_dict[venue][player]['bowled/i']=0
                    venue_player_dict[venue][player]['spin_runs_b1']=0
                    venue_player_dict[venue][player]['spin_runs_b2']=0
                    venue_player_dict[venue][player]['spin_runs_b3']=0
                    venue_player_dict[venue][player]['vbaHs_4s_1']=0
                    venue_player_dict[venue][player]['vbaHs_4s_2']=0
                    venue_player_dict[venue][player]['vbaHs_4s_3']=0
                    venue_player_dict[venue][player]['vbaHs_6s_1']=0
                    venue_player_dict[venue][player]['vbaHs_6s_2']=0
                    venue_player_dict[venue][player]['vbaHs_6s_3']=0
                    venue_player_dict[venue][player]['vbaHs_dismissals_1']=0
                    venue_player_dict[venue][player]['vbaHs_dismissals_2']=0
                    venue_player_dict[venue][player]['vbaHs_dismissals_3']=0
                    venue_player_dict[venue][player]['spin_balls_b1']=0
                    venue_player_dict[venue][player]['spin_balls_b2']=0
                    venue_player_dict[venue][player]['spin_balls_b3']=0
                    venue_player_dict[venue][player]['pace_runs_b1']=0
                    venue_player_dict[venue][player]['pace_runs_b2']=0
                    venue_player_dict[venue][player]['pace_runs_b3']=0
                    venue_player_dict[venue][player]['vbaHp_dismissals_1']=0
                    venue_player_dict[venue][player]['vbaHp_dismissals_2']=0
                    venue_player_dict[venue][player]['vbaHp_dismissals_3']=0
                    venue_player_dict[venue][player]['pace_balls_b1']=0
                    venue_player_dict[venue][player]['pace_balls_b2']=0
                    venue_player_dict[venue][player]['pace_balls_b3']=0
                    venue_player_dict[venue][player]['vbaHp_6s_1']=0
                    venue_player_dict[venue][player]['vbaHp_6s_2']=0
                    venue_player_dict[venue][player]['vbaHp_6s_3']=0
                    venue_player_dict[venue][player]['vbaHp_4s_1']=0
                    venue_player_dict[venue][player]['vbaHp_4s_2']=0
                    venue_player_dict[venue][player]['vbaHp_4s_3']=0
                    venue_player_dict[venue][player]['vbaHs_economy_1']=0
                    venue_player_dict[venue][player]['vbaHp_economy_1']=0
                    venue_player_dict[venue][player]['vbaHs_economy_2']=0
                    venue_player_dict[venue][player]['vbaHp_economy_2']=0
                    venue_player_dict[venue][player]['vbaHs_economy_3']=0
                    venue_player_dict[venue][player]['vbaHp_economy_3']=0
                    venue_player_dict[venue][player]['vbaHr_economy_1']=0
                    venue_player_dict[venue][player]['vbaHr_economy_2']=0
                    venue_player_dict[venue][player]['vbaHr_economy_3']=0
                    venue_player_dict[venue][player]['vbaHr_4s_1']=0
                    venue_player_dict[venue][player]['vbaHr_4s_2']=0
                    venue_player_dict[venue][player]['vbaHr_4s_3']=0    
                    venue_player_dict[venue][player]['vbaHr_6s_1']=0
                    venue_player_dict[venue][player]['vbaHr_6s_2']=0
                    venue_player_dict[venue][player]['vbaHr_6s_3']=0
                    venue_player_dict[venue][player]['vbaHr_dismissals_1']=0
                    venue_player_dict[venue][player]['vbaHr_dismissals_2']=0
                    venue_player_dict[venue][player]['vbaHr_dismissals_3']=0 
                    venue_player_dict[venue][player]['vbaHs_economy_agg']=0
                    venue_player_dict[venue][player]['vbaHp_economy_agg']=0         
                    venue_player_dict[venue][player]['vbaHr_economy_agg']=0
                    venue_player_dict[venue][player]['vbaHs_4s_agg']=0
                    venue_player_dict[venue][player]['vbaHp_4s_agg']=0         
                    venue_player_dict[venue][player]['vbaHr_4s_agg']=0
                    venue_player_dict[venue][player]['vbaHs_dismissals_agg']=0
                    venue_player_dict[venue][player]['vbaHp_dismissals_agg']=0         
                    venue_player_dict[venue][player]['vbaHr_dismissals_agg']=0
                    venue_player_dict[venue][player]['vbaHs_6s_agg']=0
                    venue_player_dict[venue][player]['vbaHp_6s_agg']=0         
                    venue_player_dict[venue][player]['vbaHr_6s_agg']=0
                    venue_player_dict[venue][player]['average']=0
                    venue_player_dict[venue][player]['average_4s']=0
                    venue_player_dict[venue][player]['average_6s']=0
                    venue_player_dict[venue][player]['average_balls_involved']=0
                    venue_player_dict[venue][player]['balls_involved']=0
                    venue_player_dict[venue][player]['balls']=0                                                                                

                try:
                    overall_bowler_details.index.get_loc(player)
                except:
                    overall_bowler_details.loc[player,'team'] = playing_df['Team'][playing_df['player_name']==player_name].values[0] 
                    overall_bowler_details.loc[player, 'innings'] = 0
                    overall_bowler_details.loc[player, 'runs'] = 0
                    overall_bowler_details.loc[player, 'extras'] = 0
                    overall_bowler_details.loc[player, 'balls'] = 0
                    overall_bowler_details.loc[player, '1s'] = 0
                    overall_bowler_details.loc[player, '2s'] = 0
                    overall_bowler_details.loc[player, '4s'] = 0
                    overall_bowler_details.loc[player, '6s'] = 0
                    overall_bowler_details.loc[player, 'dots'] = 0
                    overall_bowler_details.loc[player, 'maidens'] = 0
                    overall_bowler_details.loc[player, 'maidens_1'] = 0
                    overall_bowler_details.loc[player, 'maidens_2'] = 0
                    overall_bowler_details.loc[player, 'maidens_3'] = 0
                    overall_bowler_details.loc[player, 'wickets'] = 0
                    overall_bowler_details.loc[player, 'wickets_1'] = 0
                    overall_bowler_details.loc[player, 'wickets_2'] = 0
                    overall_bowler_details.loc[player, 'wickets_3'] = 0
                    overall_bowler_details.loc[player, 'bowledlbw'] = 0
                    overall_bowler_details.loc[player, 'bowledlbw_1'] = 0
                    overall_bowler_details.loc[player, 'bowledlbw_2'] = 0
                    overall_bowler_details.loc[player, 'bowledlbw_3'] = 0
                    overall_bowler_details.loc[player, 'average'] = 0
                    overall_bowler_details.loc[player, 'strike_rate'] = 0
                    overall_bowler_details.loc[player, 'economy'] = 0
                    overall_bowler_details.loc[player, '1haul'] = 0
                    overall_bowler_details.loc[player, '3haul'] = 0
                    overall_bowler_details.loc[player, '5haul'] = 0
                    overall_bowler_details.loc[player, 'balls_1'] = 0
                    overall_bowler_details.loc[player, 'balls_2'] = 0
                    overall_bowler_details.loc[player, 'balls_3'] = 0   
                if venue not in venue_dict_bowl:
                        venue_dict_bowl[venue] = {}
                        venue_dict_bowl[venue]['caught'] = 0
                        venue_dict_bowl[venue]['run_out'] = 0
                        venue_dict_bowl[venue]['stumped'] = 0
                        venue_dict_bowl[venue]['innings'] = 0
                        venue_dict_bowl[venue]['wickets'] = 0
                        venue_dict_bowl[venue]['bowledlbw'] = 0
                        venue_dict_bowl[venue]['maidens'] = 0
                        venue_dict_bowl[venue]['runs'] = 0
                        venue_dict_bowl[venue]['balls'] = 0    
                try:
                    overall_fielding_details.index.get_loc(player)
                except:
                    
                    overall_fielding_details.loc[player,'caught'] = 0
                    overall_fielding_details.loc[player,'runouts'] = 0
                    overall_fielding_details.loc[player,'stumped'] = 0
                    overall_fielding_details.loc[player,'innings'] = 0
            
                
                # PREVIOUS DATA
                if venue not in venue_dict_field:
                    venue_dict_field[venue] = {}
                    venue_dict_field[venue]['caught'] = 0
                    venue_dict_field[venue]['run_out'] = 0
                    venue_dict_field[venue]['stumped'] = 0
                    venue_dict_field[venue]['innings'] = 0                      
                count=len(inference_batsman_details)
                
                
                
                # inference_batsman_details.loc[count,'match_id'] = match_id
                inference_batsman_details.loc[count,'date'] = date
                inference_batsman_details.loc[count,'player_name'] = player
                inference_batsman_details.loc[count,'team'] = playing_df['Team'][playing_df['player_name']==player_name].values[0]
                inference_batsman_details.loc[count,'inning_no'] = playing_df['inning_no'][playing_df['player_name']==player_name].values[0]  
                inference_batsman_details.loc[count,'opposition_name'] = playing_df['Opposition'][playing_df['player_name']==player_name].values[0] 
                inference_batsman_details.loc[count,'venue_name'] = venue
                inference_batsman_details.loc[count,'batting_order'] =playing_df['lineupOrder'][playing_df['player_name']==player_name].values[0]  
                inference_batsman_details.loc[count,'innings_played'] = overall_batsman_details.loc[player,'innings'][-1]
                
                inference_batsman_details.loc[count,'previous_balls'] = overall_batsman_details.loc[player,'balls'][-1]
                inference_batsman_details.loc[count,'previous_balls_involved'] = overall_batsman_details.loc[player,'balls_involved'][-1]
                inference_batsman_details.loc[count,'previous_outs'] = overall_batsman_details.loc[player,'outs'][-1]

                inference_batsman_details.loc[count,'previous_average'] = overall_batsman_details.loc[player, 'average'][-1]
                inference_batsman_details.loc[count,'previous_strike_rate'] = overall_batsman_details.loc[player, 'strike_rate'][-1]
                inference_batsman_details.loc[count,'previous_centuries'] = overall_batsman_details.loc[player,'centuries'][-1]
                inference_batsman_details.loc[count,'previous_fifties'] = overall_batsman_details.loc[player,'fifties'][-1]
                inference_batsman_details.loc[count,'previous_thirties'] = overall_batsman_details.loc[player,'thirties'][-1]
                inference_batsman_details.loc[count,'previous_zeros'] = overall_batsman_details.loc[player,'zeros'][-1]
                inference_batsman_details.loc[count,'previous_runs'] = overall_batsman_details.loc[player,'runs'][-1]
                inference_batsman_details.loc[count,'previous_4s'] = overall_batsman_details.loc[player,'4s'][-1]
                inference_batsman_details.loc[count,'previous_6s'] = overall_batsman_details.loc[player,'6s'][-1]
                
             
                
                


                # try:
                #     if inference_batsman_details.loc[count,'T20_match_runs'] >= player_highest_score[inference_batsman_details.loc[count,'player_name']]:
                #         player_highest_score[inference_batsman_details.loc[count,'player_name']] = inference_batsman_details.loc[count,'T20_match_runs']
                # except:
                #         player_highest_score[inference_batsman_details.loc[count,'player_name']] = inference_batsman_details.loc[count,'T20_match_runs']
                # inference_batsman_details.loc[count,'T20_match_highest_score'] = player_highest_score[inference_batsman_details.loc[count,'player_name']]            
                
                # putting stuff in venue dict

                
                # inference_batsman_details.loc[count,'venue'] = 0.4262*venue_dict[player][venue]['previous_average'] + 0.2566*venue_dict[player][venue]['previous_innings']+0.1510*venue_dict[player][venue]['previous_strike_rate']+0.0787*venue_dict[player][venue]['previous_centuries']+0.0556*venue_dict[player][venue]['previous_fifties']+0.0328*venue_dict[player][venue]['high_score']  
                
                # inference_batsman_details.loc[count,'venue_avg']=venue_dict[player][venue]['average']
                try: 
                 if opposition_dict[player][opposition]['balls'] != 0:
                    
                      opposition_dict[player][opposition]['strike_rate'] = opposition_dict[player][opposition]['runs'] * 100 / opposition_dict[player][opposition]['balls']
                except:
                        pass  
                else:
                    opposition_dict[player][opposition]['strike_rate'] = opposition_dict[player][opposition]['runs']
                if opposition_dict[player][opposition]['wickets'] == 0:
                    opposition_dict[player][opposition]['average'] = opposition_dict[player][opposition]['runs']
                else:
                    
                    opposition_dict[player][opposition]['average'] = int(opposition_dict[player][opposition]['runs']) / int(opposition_dict[player][opposition]['wickets'])    
                inference_batsman_details.loc[count,'opposition'] = 0.4262*opposition_dict[player][opposition]['average'] + 0.2566*opposition_dict[player][opposition]['innings']+0.1510*opposition_dict[player][opposition]['strike_rate']+0.0787*opposition_dict[player][opposition]['centuries']+0.0556*opposition_dict[player][opposition]['fifties']+0.0328*opposition_dict[player][opposition]['zeros']

                    
                print(inference_batsman_details.loc[count,'previous_average']*2)
                inference_batsman_details.loc[count,'consistency']=0.4262*inference_batsman_details.loc[count,'previous_average']+0.2566*inference_batsman_details.loc[count,'innings_played']+0.1510*inference_batsman_details.loc[count,'previous_strike_rate']+0.0787*inference_batsman_details.loc[count,'previous_centuries']+0.0556*inference_batsman_details.loc[count,'previous_fifties'] - 0.0328*inference_batsman_details.loc[count,'previous_zeros']
                indices = inference_batsman_details[(inference_batsman_details['player_name'] == player) & (inference_batsman_details.index < count)].index
                if len(indices) == 0:
                    inference_batsman_details.loc[count,'form'] = 0
                else:
                    cur = inference_batsman_details.loc[count]
                    old = inference_batsman_details.loc[count-min(10, len(indices))]
                    try:
                        avg = (cur['previous_runs'] - old['previous_runs']) / (cur['previous_outs'] - old['previous_outs'])
                    except:
                        avg = 0
                    try:
                        sr = (cur['previous_runs'] - old['previous_runs']) * 100 / (cur['previous_balls'] - old['previous_balls'])
                    except:
                        sr = 0
                    cent = cur['previous_centuries'] - old['previous_centuries']
                    fift = cur['previous_fifties'] - old['previous_fifties']
                    zer = cur['previous_zeros'] - old['previous_zeros']
                    inn = min(10, len(indices))
                    inference_batsman_details.loc[count,'form']=0.4262*avg+0.2566*inn+0.1510*sr+0.0787*cent+0.0556*fift-0.0328*(zer)
                
                # inference_batsman_details.loc[count,'form']=0.4262*tokenize(inference_batsman_details.loc[count,'previous_average'],{0:1,10:2,20:3,30:4,40:5})+0.2566*tokenize(inference_batsman_details.loc[count,'innings_played'],{1:1,5:2,10:3,12:4,15:5})+0.1510*tokenize(inference_batsman_details.loc[count,'previous_strike_rate'],{0:1,50:2,60:3,80:4,100:5})+0.0787*tokenize(inference_batsman_details.loc[count,'previous_centuries'],{0:1,2:2,3:3,4:4,5:5})+0.0556*tokenize(inference_batsman_details.loc[count,'previous_fifties'],{0:1,3:2,5:3,7:4,10:5})-0.0328*tokenize(inference_batsman_details.loc[count,'previous_zeros'],{0:1,2:2,3:3,4:4,5:5})
                
                # inference_batsman_details.loc[count,'match_runs'] = inference_batsman_details.loc[count,'runs']
                # inference_batsman_details.loc[count,'match_4s'] = inference_batsman_details.loc[count,'4s']
                # inference_batsman_details.loc[count,'match_6s'] = inference_batsman_details.loc[count,'6s']
                # inference_batsman_details.loc[count,'match_strike_rate'] = (player_details.loc[player,'runs'] * 100 / player_details.loc[player,'balls']) if player_details.loc[player,'balls'] != 0 else 0
                # bonus = 0
                # if inference_batsman_details.loc[count,'match_runs'] >= 100:
                #     bonus = 16
                # elif inference_batsman_details.loc[count,'match_runs'] >= 50:
                #     bonus = 8
                # elif inference_batsman_details.loc[count,'match_runs'] >= 30:
                #     bonus = 4
                # elif inference_batsman_details.loc[count,'match_runs'] == 0:
                #     bonus = -2
                # sr = 0
                # try:
                #     sr = player_details.loc[player,'runs'] * 100 / player_details.loc[player,'balls']
                # except:
                #     pass
                # if player_details.loc[player,'balls'] >= 10:
                #     if sr >= 170:
                #         bonus += 6
                #     elif sr >= 150:
                #         bonus += 4
                #     elif sr >= 130:
                #         bonus += 2
                #     if sr < 50:
                #         bonus -= 6
                #     elif sr < 60:
                #         bonus -= 4
                #     elif sr < 70:
                #         bonus -= 2
                # inference_batsman_details.loc[count,'match_fantasy_points'] = inference_batsman_details.loc[count,'match_runs'] + inference_batsman_details.loc[count,'match_4s'] + 2*inference_batsman_details.loc[count,'match_6s'] + bonus
            

                

                inference_batsman_details.loc[count,'hist_spin_runs_b1']=sum(overall_batsman_details.loc[player,'spin_runs_b1'])
                inference_batsman_details.loc[count,'hist_spin_runs_b2']=sum(overall_batsman_details.loc[player,'spin_runs_b2'])
                inference_batsman_details.loc[count,'hist_spin_runs_b3']=sum(overall_batsman_details.loc[player,'spin_runs_b3'])
                inference_batsman_details.loc[count,'hist_pace_runs_b1']=sum(overall_batsman_details.loc[player,'pace_runs_b1'])
                inference_batsman_details.loc[count,'hist_pace_runs_b2']=sum(overall_batsman_details.loc[player,'pace_runs_b2'])
                inference_batsman_details.loc[count,'hist_pace_runs_b3']=sum(overall_batsman_details.loc[player,'pace_runs_b3'])
                inference_batsman_details.loc[count,'hist_spin_balls_b1']=sum(overall_batsman_details.loc[player,'spin_balls_b1'])
                inference_batsman_details.loc[count,'hist_spin_balls_b2']=sum(overall_batsman_details.loc[player,'spin_balls_b2'])
                inference_batsman_details.loc[count,'hist_spin_balls_b3']=sum(overall_batsman_details.loc[player,'spin_balls_b3'])
                inference_batsman_details.loc[count,'hist_pace_balls_b1']=sum(overall_batsman_details.loc[player,'pace_balls_b1'])
                inference_batsman_details.loc[count,'hist_pace_balls_b2']=sum(overall_batsman_details.loc[player,'pace_balls_b2'])
                inference_batsman_details.loc[count,'hist_pace_balls_b3']=sum(overall_batsman_details.loc[player,'pace_balls_b3'])
                inference_batsman_details.loc[count,'tbaHs_dismissals_1']=sum(overall_batsman_details.loc[player,'spin_wickets_b1'])
                inference_batsman_details.loc[count,'tbaHs_dismissals_2']=sum(overall_batsman_details.loc[player,'spin_wickets_b2'])
                inference_batsman_details.loc[count,'tbaHs_dismissals_3']=sum(overall_batsman_details.loc[player,'spin_wickets_b3'])
                inference_batsman_details.loc[count,'tbaHp_dismissals_1']=sum(overall_batsman_details.loc[player,'pace_wickets_b1'])
                inference_batsman_details.loc[count,'tbaHp_dismissals_2']=sum(overall_batsman_details.loc[player,'pace_wickets_b2'])
                inference_batsman_details.loc[count,'tbaHp_dismissals_3']=sum(overall_batsman_details.loc[player,'pace_wickets_b3'])
            
                inference_batsman_details.loc[count,'tbaHs_6s_1']=sum(overall_batsman_details.loc[player,'spin_6s_b1'])
                inference_batsman_details.loc[count,'tbaHs_6s_2']=sum(overall_batsman_details.loc[player,'spin_6s_b2'])
                inference_batsman_details.loc[count,'tbaHs_6s_3']=sum(overall_batsman_details.loc[player,'spin_6s_b3'])
                inference_batsman_details.loc[count,'tbaHp_6s_1']=sum(overall_batsman_details.loc[player,'pace_6s_b1'])
                inference_batsman_details.loc[count,'tbaHp_6s_2']=sum(overall_batsman_details.loc[player,'pace_6s_b2'])
                inference_batsman_details.loc[count,'tbaHp_6s_3']=sum(overall_batsman_details.loc[player,'pace_6s_b3'])
                inference_batsman_details.loc[count,'tbaHs_4s_1']=sum(overall_batsman_details.loc[player,'spin_4s_b1'])
                inference_batsman_details.loc[count,'tbaHs_4s_2']=sum(overall_batsman_details.loc[player,'spin_4s_b2'])
                inference_batsman_details.loc[count,'tbaHs_4s_3']=sum(overall_batsman_details.loc[player,'spin_4s_b3'])
                inference_batsman_details.loc[count,'tbaHp_4s_1']=sum(overall_batsman_details.loc[player,'pace_4s_b1'])
                inference_batsman_details.loc[count,'tbaHp_4s_2']=sum(overall_batsman_details.loc[player,'pace_4s_b2'])
                inference_batsman_details.loc[count,'tbaHp_4s_3']=sum(overall_batsman_details.loc[player,'pace_4s_b3'])
                inference_batsman_details.loc[count,'tbaHs_economy_1']=inference_batsman_details.loc[count,'hist_spin_runs_b1']/(inference_batsman_details.loc[count,'hist_spin_balls_b1']+0.1)
                inference_batsman_details.loc[count,'tbaHs_economy_2']=inference_batsman_details.loc[count,'hist_spin_runs_b2']/(inference_batsman_details.loc[count,'hist_spin_balls_b2']+0.1)
                inference_batsman_details.loc[count,'tbaHs_economy_3']=inference_batsman_details.loc[count,'hist_spin_runs_b3']/(inference_batsman_details.loc[count,'hist_spin_balls_b3']+0.1)
                inference_batsman_details.loc[count,'tbaHp_economy_1']=inference_batsman_details.loc[count,'hist_pace_runs_b1']/(inference_batsman_details.loc[count,'hist_pace_balls_b1']+0.1)
                inference_batsman_details.loc[count,'tbaHp_economy_2']=inference_batsman_details.loc[count,'hist_pace_runs_b2']/(inference_batsman_details.loc[count,'hist_pace_balls_b2']+0.1)
                inference_batsman_details.loc[count,'tbaHp_economy_3']=inference_batsman_details.loc[count,'hist_pace_runs_b3']/(inference_batsman_details.loc[count,'hist_pace_balls_b3']+0.1)
                inference_batsman_details.loc[count,'tbaHr_economy_1']=inference_batsman_details.loc[count,'tbaHs_economy_1']/(inference_batsman_details.loc[count,'tbaHp_economy_1']+0.1)
                inference_batsman_details.loc[count,'tbaHr_economy_2']=inference_batsman_details.loc[count,'tbaHs_economy_2']/(inference_batsman_details.loc[count,'tbaHp_economy_2']+0.1)
                inference_batsman_details.loc[count,'tbaHr_economy_3']=inference_batsman_details.loc[count,'tbaHs_economy_3']/(inference_batsman_details.loc[count,'tbaHp_economy_3']+0.1)

                inference_batsman_details.loc[count,'tbaHr_4s_1']=inference_batsman_details.loc[count,'tbaHs_4s_1']/(inference_batsman_details.loc[count,'tbaHp_4s_1']+0.1)
                inference_batsman_details.loc[count,'tbaHr_4s_2']=inference_batsman_details.loc[count,'tbaHs_4s_2']/(inference_batsman_details.loc[count,'tbaHp_4s_2']+0.1)
                inference_batsman_details.loc[count,'tbaHr_4s_3']=inference_batsman_details.loc[count,'tbaHs_4s_3']/(inference_batsman_details.loc[count,'tbaHp_4s_3']+0.1)
                inference_batsman_details.loc[count,'tbaHr_6s_1']=inference_batsman_details.loc[count,'tbaHs_6s_1']/(inference_batsman_details.loc[count,'tbaHp_6s_1']+0.1)
                inference_batsman_details.loc[count,'tbaHr_6s_2']=inference_batsman_details.loc[count,'tbaHs_6s_2']/(inference_batsman_details.loc[count,'tbaHp_6s_2']+0.1)
                inference_batsman_details.loc[count,'tbaHr_6s_3']=inference_batsman_details.loc[count,'tbaHs_6s_3']/(inference_batsman_details.loc[count,'tbaHp_6s_3']+0.1)
                inference_batsman_details.loc[count,'tbaHr_dismissals_1']=inference_batsman_details.loc[count,'tbaHs_dismissals_1']/(inference_batsman_details.loc[count,'tbaHp_dismissals_1']+0.1)
                inference_batsman_details.loc[count,'tbaHr_dismissals_2']=inference_batsman_details.loc[count,'tbaHs_dismissals_2']/(inference_batsman_details.loc[count,'tbaHp_dismissals_2']+0.1)
                inference_batsman_details.loc[count,'tbaHr_dismissals_3']=inference_batsman_details.loc[count,'tbaHs_dismissals_3']/(inference_batsman_details.loc[count,'tbaHp_dismissals_3']+0.1)
                inference_batsman_details.loc[count,'tbaHs_economy_agg']=sum([inference_batsman_details.loc[count,i] for i in ['hist_spin_runs_b1','hist_spin_runs_b2','hist_spin_runs_b3']])/(sum([inference_batsman_details.loc[count,i] for i in ['hist_spin_balls_b1','hist_spin_balls_b2','hist_spin_balls_b3']])+0.1)
                inference_batsman_details.loc[count,'tbaHp_economy_agg']=sum([inference_batsman_details.loc[count,i] for i in ['hist_pace_runs_b1','hist_pace_runs_b2','hist_pace_runs_b3']])/(sum([inference_batsman_details.loc[count,i] for i in ['hist_pace_balls_b1','hist_pace_balls_b2','hist_pace_balls_b3']])+0.1)
                inference_batsman_details.loc[count,'tbaHr_economy_agg']=inference_batsman_details.loc[count,'tbaHs_economy_agg']/inference_batsman_details.loc[count,'tbaHp_economy_agg']
                inference_batsman_details.loc[count,'tbaHs_4s_agg']=sum([inference_batsman_details.loc[count,i] for i in ['tbaHs_4s_1','tbaHs_4s_2','tbaHs_4s_3']])
                inference_batsman_details.loc[count,'tbaHp_4s_agg']=sum([inference_batsman_details.loc[count,i] for i in ['tbaHp_4s_1','tbaHp_4s_2','tbaHp_4s_3']])
                inference_batsman_details.loc[count,'tbaHr_4s_agg']=inference_batsman_details.loc[count,'tbaHs_4s_agg']/(inference_batsman_details.loc[count,'tbaHp_4s_agg']+0.01)
                inference_batsman_details.loc[count,'tbaHs_6s_agg']=sum([inference_batsman_details.loc[count,i] for i in ['tbaHs_6s_1','tbaHs_6s_2','tbaHs_6s_3']])
                inference_batsman_details.loc[count,'tbaHp_6s_agg']=sum([inference_batsman_details.loc[count,i] for i in ['tbaHp_6s_1','tbaHp_6s_2','tbaHp_6s_3']])
                inference_batsman_details.loc[count,'tbaHr_6s_agg']=inference_batsman_details.loc[count,'tbaHs_6s_agg']/(inference_batsman_details.loc[count,'tbaHp_6s_agg']+0.01)
                inference_batsman_details.loc[count,'tbaHs_dismissals_agg']=sum([inference_batsman_details.loc[count,i] for i in ['tbaHs_dismissals_1','tbaHs_dismissals_2','tbaHs_dismissals_3']])
                inference_batsman_details.loc[count,'tbaHp_dismissals_agg']=sum([inference_batsman_details.loc[count,i] for i in ['tbaHp_dismissals_1','tbaHp_dismissals_2','tbaHp_dismissals_3']])
                inference_batsman_details.loc[count,'tbaHr_dismissals_agg']=inference_batsman_details.loc[count,'tbaHs_dismissals_agg']/(inference_batsman_details.loc[count,'tbaHp_dismissals_agg']+0.01)
                
                

                inference_batsman_details.loc[count,'form_spin_runs_b1']=sum(overall_batsman_details.loc[player,'spin_runs_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_spin_runs_b2']=sum(overall_batsman_details.loc[player,'spin_runs_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_spin_runs_b3']=sum(overall_batsman_details.loc[player,'spin_runs_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_pace_runs_b1']=sum(overall_batsman_details.loc[player,'pace_runs_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_pace_runs_b2']=sum(overall_batsman_details.loc[player,'pace_runs_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_pace_runs_b3']=sum(overall_batsman_details.loc[player,'pace_runs_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_spin_balls_b1']=sum(overall_batsman_details.loc[player,'spin_balls_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_spin_balls_b2']=sum(overall_batsman_details.loc[player,'spin_balls_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_spin_balls_b3']=sum(overall_batsman_details.loc[player,'spin_balls_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_pace_balls_b1']=sum(overall_batsman_details.loc[player,'pace_balls_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_pace_balls_b2']=sum(overall_batsman_details.loc[player,'pace_balls_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'form_pace_balls_b3']=sum(overall_batsman_details.loc[player,'pace_balls_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_dismissals_1']=sum(overall_batsman_details.loc[player,'spin_wickets_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_dismissals_2']=sum(overall_batsman_details.loc[player,'spin_wickets_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_dismissals_3']=sum(overall_batsman_details.loc[player,'spin_wickets_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_dismissals_1']=sum(overall_batsman_details.loc[player,'pace_wickets_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_dismissals_2']=sum(overall_batsman_details.loc[player,'pace_wickets_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_dismissals_3']=sum(overall_batsman_details.loc[player,'pace_wickets_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_6s_1']=sum(overall_batsman_details.loc[player,'spin_6s_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_6s_2']=sum(overall_batsman_details.loc[player,'spin_6s_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_6s_3']=sum(overall_batsman_details.loc[player,'spin_6s_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_6s_1']=sum(overall_batsman_details.loc[player,'pace_6s_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_6s_2']=sum(overall_batsman_details.loc[player,'pace_6s_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_6s_3']=sum(overall_batsman_details.loc[player,'pace_6s_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_4s_1']=sum(overall_batsman_details.loc[player,'spin_4s_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_4s_2']=sum(overall_batsman_details.loc[player,'spin_4s_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHs_4s_3']=sum(overall_batsman_details.loc[player,'spin_4s_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_4s_1']=sum(overall_batsman_details.loc[player,'pace_4s_b1'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_4s_2']=sum(overall_batsman_details.loc[player,'pace_4s_b2'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])
                inference_batsman_details.loc[count,'fbaHp_4s_3']=sum(overall_batsman_details.loc[player,'pace_4s_b3'][-1*min(10,len(overall_batsman_details.loc[player,'spin_runs_b1'])):])


                inference_batsman_details.loc[count,'fbaHs_economy_1']=inference_batsman_details.loc[count,'form_spin_runs_b1']/(inference_batsman_details.loc[count,'form_spin_balls_b1']+0.1)
                inference_batsman_details.loc[count,'fbaHs_economy_2']=inference_batsman_details.loc[count,'form_spin_runs_b2']/(inference_batsman_details.loc[count,'form_spin_balls_b2']+0.1)
                inference_batsman_details.loc[count,'fbaHs_economy_3']=inference_batsman_details.loc[count,'form_spin_runs_b3']/(inference_batsman_details.loc[count,'form_spin_balls_b3']+0.1)
                inference_batsman_details.loc[count,'fbaHp_economy_1']=inference_batsman_details.loc[count,'form_pace_runs_b1']/(inference_batsman_details.loc[count,'form_pace_balls_b1']+0.1)
                inference_batsman_details.loc[count,'fbaHp_economy_2']=inference_batsman_details.loc[count,'form_pace_runs_b2']/(inference_batsman_details.loc[count,'form_pace_balls_b2']+0.1)
                inference_batsman_details.loc[count,'fbaHp_economy_3']=inference_batsman_details.loc[count,'form_pace_runs_b3']/(inference_batsman_details.loc[count,'form_pace_balls_b3']+0.1)
                inference_batsman_details.loc[count,'fbaHr_economy_1']=inference_batsman_details.loc[count,'fbaHs_economy_1']/(inference_batsman_details.loc[count,'fbaHp_economy_1']+0.1)
                inference_batsman_details.loc[count,'fbaHr_economy_2']=inference_batsman_details.loc[count,'fbaHs_economy_2']/(inference_batsman_details.loc[count,'fbaHp_economy_2']+0.1)
                inference_batsman_details.loc[count,'fbaHr_economy_3']=inference_batsman_details.loc[count,'fbaHs_economy_3']/(inference_batsman_details.loc[count,'fbaHp_economy_3']+0.1)

                inference_batsman_details.loc[count,'fbaHr_4s_1']=inference_batsman_details.loc[count,'fbaHs_4s_1']/(inference_batsman_details.loc[count,'fbaHp_4s_1']+0.1)
                inference_batsman_details.loc[count,'fbaHr_4s_2']=inference_batsman_details.loc[count,'fbaHs_4s_2']/(inference_batsman_details.loc[count,'fbaHp_4s_2']+0.1)
                inference_batsman_details.loc[count,'fbaHr_4s_3']=inference_batsman_details.loc[count,'fbaHs_4s_3']/(inference_batsman_details.loc[count,'fbaHp_4s_3']+0.1)
                inference_batsman_details.loc[count,'fbaHr_6s_1']=inference_batsman_details.loc[count,'fbaHs_6s_1']/(inference_batsman_details.loc[count,'fbaHp_6s_1']+0.1)
                inference_batsman_details.loc[count,'fbaHr_6s_2']=inference_batsman_details.loc[count,'fbaHs_6s_2']/(inference_batsman_details.loc[count,'fbaHp_6s_2']+0.1)
                inference_batsman_details.loc[count,'fbaHr_6s_3']=inference_batsman_details.loc[count,'fbaHs_6s_3']/(inference_batsman_details.loc[count,'fbaHp_6s_3']+0.1)
                inference_batsman_details.loc[count,'fbaHr_dismissals_1']=inference_batsman_details.loc[count,'fbaHs_dismissals_1']/(inference_batsman_details.loc[count,'fbaHp_dismissals_1']+0.1)
                inference_batsman_details.loc[count,'fbaHr_dismissals_2']=inference_batsman_details.loc[count,'fbaHs_dismissals_2']/(inference_batsman_details.loc[count,'fbaHp_dismissals_2']+0.1)
                inference_batsman_details.loc[count,'fbaHr_dismissals_3']=inference_batsman_details.loc[count,'fbaHs_dismissals_3']/(inference_batsman_details.loc[count,'fbaHp_dismissals_3']+0.1)
                inference_batsman_details.loc[count,'fbaHs_economy_agg']=sum([inference_batsman_details.loc[count,i] for i in ['form_spin_runs_b1','form_spin_runs_b2','form_spin_runs_b3']])/(sum([inference_batsman_details.loc[count,i] for i in ['form_spin_balls_b1','form_spin_balls_b2','form_spin_balls_b3']])+0.1)
                inference_batsman_details.loc[count,'fbaHp_economy_agg']=sum([inference_batsman_details.loc[count,i] for i in ['form_pace_runs_b1','form_pace_runs_b2','form_pace_runs_b3']])/(sum([inference_batsman_details.loc[count,i] for i in ['form_pace_balls_b1','form_pace_balls_b2','form_pace_balls_b3']])+0.1)
                inference_batsman_details.loc[count,'fbaHr_economy_agg']=inference_batsman_details.loc[count,'fbaHs_economy_agg']/inference_batsman_details.loc[count,'fbaHp_economy_agg']
                inference_batsman_details.loc[count,'fbaHs_4s_agg']=sum([inference_batsman_details.loc[count,i] for i in ['fbaHs_4s_1','fbaHs_4s_2','fbaHs_4s_3']])
                inference_batsman_details.loc[count,'fbaHp_4s_agg']=sum([inference_batsman_details.loc[count,i] for i in ['fbaHp_4s_1','fbaHp_4s_2','fbaHp_4s_3']])
                inference_batsman_details.loc[count,'fbaHr_4s_agg']=inference_batsman_details.loc[count,'fbaHs_4s_agg']/inference_batsman_details.loc[count,'fbaHp_4s_agg']
                inference_batsman_details.loc[count,'fbaHs_6s_agg']=sum([inference_batsman_details.loc[count,i] for i in ['fbaHs_6s_1','fbaHs_6s_2','fbaHs_6s_3']])
                inference_batsman_details.loc[count,'fbaHp_6s_agg']=sum([inference_batsman_details.loc[count,i] for i in ['fbaHp_6s_1','fbaHp_6s_2','fbaHp_6s_3']])
                inference_batsman_details.loc[count,'fbaHr_6s_agg']=inference_batsman_details.loc[count,'fbaHs_6s_agg']/inference_batsman_details.loc[count,'fbaHp_6s_agg']
                inference_batsman_details.loc[count,'fbaHs_dismissals_agg']=sum([inference_batsman_details.loc[count,i] for i in ['fbaHs_dismissals_1','fbaHs_dismissals_2','fbaHs_dismissals_3']])
                inference_batsman_details.loc[count,'fbaHp_dismissals_agg']=sum([inference_batsman_details.loc[count,i] for i in ['fbaHp_dismissals_1','fbaHp_dismissals_2','fbaHp_dismissals_3']])
                inference_batsman_details.loc[count,'fbaHr_dismissals_agg']=inference_batsman_details.loc[count,'fbaHs_dismissals_agg']/inference_batsman_details.loc[count,'fbaHp_dismissals_agg']
                inference_batsman_details.loc[count,'spin_runs_b1']=overall_batsman_details.loc[player,'spin_runs_b1'][-1]
                inference_batsman_details.loc[count,'spin_runs_b2']=overall_batsman_details.loc[player,'spin_runs_b2'][-1]
                inference_batsman_details.loc[count,'spin_runs_b3']=overall_batsman_details.loc[player,'spin_runs_b3'][-1]
                inference_batsman_details.loc[count,'pace_runs_b1']=overall_batsman_details.loc[player,'pace_runs_b1'][-1]
                inference_batsman_details.loc[count,'pace_runs_b2']=overall_batsman_details.loc[player,'pace_runs_b2'][-1]
                inference_batsman_details.loc[count,'pace_runs_b3']=overall_batsman_details.loc[player,'pace_runs_b3'][-1]
                inference_batsman_details.loc[count,'spin_balls_b1']=overall_batsman_details.loc[player,'spin_balls_b1'][-1]
                inference_batsman_details.loc[count,'spin_balls_b2']=overall_batsman_details.loc[player,'spin_balls_b2'][-1]
                inference_batsman_details.loc[count,'spin_balls_b3']=overall_batsman_details.loc[player,'spin_balls_b3'][-1]
                inference_batsman_details.loc[count,'pace_balls_b1']=overall_batsman_details.loc[player,'pace_balls_b1'][-1]
                inference_batsman_details.loc[count,'pace_balls_b2']=overall_batsman_details.loc[player,'pace_balls_b2'][-1]
                inference_batsman_details.loc[count,'pace_balls_b3']=overall_batsman_details.loc[player,'pace_balls_b3'][-1]
                inference_batsman_details.loc[count,'spin_wickets_b1']=overall_batsman_details.loc[player,'spin_wickets_b1'][-1]
                inference_batsman_details.loc[count,'spin_wickets_b2']=overall_batsman_details.loc[player,'spin_wickets_b2'][-1]
                inference_batsman_details.loc[count,'spin_wickets_b3']=overall_batsman_details.loc[player,'spin_wickets_b3'][-1]
                inference_batsman_details.loc[count,'pace_wickets_b1']=overall_batsman_details.loc[player,'pace_wickets_b1'][-1]
                inference_batsman_details.loc[count,'pace_wickets_b2']=overall_batsman_details.loc[player,'pace_wickets_b2'][-1]
                inference_batsman_details.loc[count,'pace_wickets_b3']=overall_batsman_details.loc[player,'pace_wickets_b3'][-1]
                inference_batsman_details.loc[count,'spin_6s_b1']=overall_batsman_details.loc[player,'spin_6s_b1'][-1]
                inference_batsman_details.loc[count,'spin_6s_b2']=overall_batsman_details.loc[player,'spin_6s_b2'][-1]
                inference_batsman_details.loc[count,'spin_6s_b3']=overall_batsman_details.loc[player,'spin_6s_b3'][-1]
                inference_batsman_details.loc[count,'pace_6s_b1']=overall_batsman_details.loc[player,'pace_6s_b1'][-1]
                inference_batsman_details.loc[count,'pace_6s_b2']=overall_batsman_details.loc[player,'pace_6s_b2'][-1]
                inference_batsman_details.loc[count,'pace_6s_b3']=overall_batsman_details.loc[player,'pace_6s_b3'][-1]
                inference_batsman_details.loc[count,'spin_4s_b1']=overall_batsman_details.loc[player,'spin_4s_b1'][-1]
                inference_batsman_details.loc[count,'spin_4s_b2']=overall_batsman_details.loc[player,'spin_4s_b2'][-1]
                inference_batsman_details.loc[count,'spin_4s_b3']=overall_batsman_details.loc[player,'spin_4s_b3'][-1]
                inference_batsman_details.loc[count,'pace_4s_b1']=overall_batsman_details.loc[player,'pace_4s_b1'][-1]
                inference_batsman_details.loc[count,'pace_4s_b2']=overall_batsman_details.loc[player,'pace_4s_b2'][-1]
                inference_batsman_details.loc[count,'pace_4s_b3']=overall_batsman_details.loc[player,'pace_4s_b3'][-1]
               
                #print(list(venue_player_dict[venue][player].keys()))
                #print(list(venue_dict[venue].keys()))
                venue_player_df.loc[len(venue_player_df)] = pd.DataFrame([venue_player_dict[venue][player]]).iloc[0]

                # venue_df.loc[len(venue_df)]=pd.DataFrame([venue_dict[venue]]).iloc[0]
                
                
                
                count=len(inference_bowler_details)
               
                inference_bowler_details.loc[count,'date'] = date
                inference_bowler_details.loc[count,'player_name'] = player
                # put player_id here
                # inference_bowler_details.loc[count,'team_name'] = player_details.loc[player,'team']
                # inference_bowler_details.loc[count,'opposition_name'] = player_details.loc[player,'opposition_name']
                inference_bowler_details.loc[count,'venue_name'] = venue
                inference_bowler_details.loc[count,'innings_played'] = overall_bowler_details.loc[player,'innings'] 
                inference_bowler_details.loc[count,'previous_balls_involved'] = overall_bowler_details.loc[player, 'dots']
                inference_bowler_details.loc[count,'previous_average'] = overall_bowler_details.loc[player, 'average']
                inference_bowler_details.loc[count,'previous_strike_rate'] = overall_bowler_details.loc[player, 'strike_rate']
                inference_bowler_details.loc[count,'previous_economy'] = overall_bowler_details.loc[player, 'economy']
                inference_bowler_details.loc[count,'previous_1haul'] = overall_bowler_details.loc[player, '1haul']
                inference_bowler_details.loc[count,'previous_3haul'] = overall_bowler_details.loc[player, '3haul']
                inference_bowler_details.loc[count,'previous_5haul'] = overall_bowler_details.loc[player, '5haul']
                inference_batsman_details.loc[count,'inning_no'] =1- playing_df['inning_no'][playing_df['player_name']==player_name].values[0]  
                inference_bowler_details.loc[count,'previous_wickets'] = overall_bowler_details.loc[player, 'wickets']
                inference_bowler_details.loc[count,'previous_wickets_1'] = overall_bowler_details.loc[player, 'wickets_1']
                inference_bowler_details.loc[count,'previous_wickets_2'] = overall_bowler_details.loc[player, 'wickets_2']
                inference_bowler_details.loc[count,'previous_wickets_3'] = overall_bowler_details.loc[player, 'wickets_3']
                inference_bowler_details.loc[count,'previous_maidens'] = overall_bowler_details.loc[player, 'maidens']
                inference_bowler_details.loc[count,'previous_maidens_1'] = overall_bowler_details.loc[player, 'maidens_1']
                inference_bowler_details.loc[count,'previous_maidens_2'] = overall_bowler_details.loc[player, 'maidens_2']
                inference_bowler_details.loc[count,'previous_maidens_3'] = overall_bowler_details.loc[player, 'maidens_3']
                inference_bowler_details.loc[count,'previous_bowledlbw'] = overall_bowler_details.loc[player, 'bowledlbw']
                inference_bowler_details.loc[count,'previous_bowledlbw_1'] = overall_bowler_details.loc[player, 'bowledlbw_1']
                inference_bowler_details.loc[count,'previous_bowledlbw_2'] = overall_bowler_details.loc[player, 'bowledlbw_2']
                inference_bowler_details.loc[count,'previous_bowledlbw_3'] = overall_bowler_details.loc[player, 'bowledlbw_3']
                

                
             

                inference_bowler_details.loc[count,'T20_match_strike_rate_1'] = (overall_bowler_details.loc[player, 'balls_1'] / overall_bowler_details.loc[player, 'wickets_1']) if overall_bowler_details.loc[player, 'wickets_1'] != 0 else 0
                inference_bowler_details.loc[count,'T20_match_strike_rate_2'] = (overall_bowler_details.loc[player, 'balls_2'] / overall_bowler_details.loc[player, 'wickets_2']) if overall_bowler_details.loc[player, 'wickets_2'] != 0 else 0
                inference_bowler_details.loc[count,'T20_match_strike_rate_3'] = (overall_bowler_details.loc[player, 'balls_3'] / overall_bowler_details.loc[player, 'wickets_3']) if overall_bowler_details.loc[player, 'wickets_3'] != 0 else 0  
                inference_bowler_details_venue.loc[count,'venue_innings'] = venue_dict_bowl[venue]['innings']
                inference_bowler_details_venue.loc[count,'venue_dismissals'] = venue_dict_bowl[venue]['wickets'] / (venue_dict_bowl[venue]['innings'] if venue_dict_bowl[venue]['innings'] != 0 else 1)
                if venue_dict_bowl[venue]['balls'] != 0:
                    inference_bowler_details_venue.loc[count,'venue_economy'] = venue_dict_bowl[venue]['runs']*6 / venue_dict_bowl[venue]['balls']
                else:
                    inference_bowler_details_venue.loc[count,'venue_economy'] = 0
                if venue_dict_bowl[venue]['wickets'] != 0:
                    inference_bowler_details_venue.loc[count,'venue_average'] = venue_dict_bowl[venue]['runs'] / venue_dict_bowl[venue]['wickets']
                else:
                    inference_bowler_details_venue.loc[count,'venue_average'] = venue_dict_bowl[venue]['runs'] 
                inference_bowler_details_venue.loc[count,'venue_maidens'] = venue_dict_bowl[venue]['maidens'] / (venue_dict_bowl[venue]['innings'] if venue_dict_bowl[venue]['innings'] != 0 else 1)
                inference_bowler_details_venue.loc[count,'venue_bowledlbw'] = venue_dict_bowl[venue]['bowledlbw'] / (venue_dict_bowl[venue]['innings'] if venue_dict_bowl[venue]['innings'] != 0 else 1)                
               
                ours = match_bowler_details[(match_bowler_details['player_name'] == player)]

                if ours.empty:
                    inference_bowler_details.loc[count, 'tboDa_economy_1'] = 0
                    inference_bowler_details.loc[count, 'tboDa_economy_2'] = 0
                    inference_bowler_details.loc[count, 'tboDa_economy_3'] = 0
                    inference_bowler_details.loc[count, 'tboDa_economy_agg'] = 0
                    inference_bowler_details.loc[count, 'tboDa_dismissals_1'] = 0
                    inference_bowler_details.loc[count, 'tboDa_dismissals_2'] = 0
                    inference_bowler_details.loc[count, 'tboDa_dismissals_3'] = 0
                    inference_bowler_details.loc[count, 'tboDa_dismissals_agg'] = 0
                    inference_bowler_details.loc[count, 'tboDa_bowledlbw_1'] =  0
                    inference_bowler_details.loc[count, 'tboDa_bowledlbw_2'] =  0
                    inference_bowler_details.loc[count, 'tboDa_bowledlbw_3'] =  0
                    inference_bowler_details.loc[count, 'tboDa_bowledlbw_agg'] =  0
                    inference_bowler_details.loc[count, 'tboDa_maidens_1'] =  0
                    inference_bowler_details.loc[count, 'tboDa_maidens_2'] =  0
                    inference_bowler_details.loc[count, 'tboDa_maidens_3'] =  0
                    inference_bowler_details.loc[count, 'tboDa_maidens_agg'] =  0
                else:
                    inference_bowler_details.loc[count, 'tboDa_economy_1'] = sum(ours['T20_match_runs_1']) * 6 / (sum(ours['T20_match_balls_1']) if sum(ours['T20_match_balls_1']) != 0 else 1)
                    inference_bowler_details.loc[count, 'tboDa_economy_2'] = sum(ours['T20_match_runs_2']) * 6 / (sum(ours['T20_match_balls_2']) if sum(ours['T20_match_balls_2']) != 0 else 1)
                    inference_bowler_details.loc[count, 'tboDa_economy_3'] = sum(ours['T20_match_runs_3']) * 6 / (sum(ours['T20_match_balls_3']) if sum(ours['T20_match_balls_3']) != 0 else 1)
                    #inference_bowler_details.loc[count, 'tboDa_economy_agg'] = sum(ours['tboDa_economy_agg']) * 6 / (sum(ours['balls']) if sum(ours['balls']) != 0 else 1)
                    inference_bowler_details.loc[count, 'tboDa_dismissals_1'] =  ours['T20_match_wickets_1'].mean()
                    inference_bowler_details.loc[count, 'tboDa_dismissals_2'] =  ours['T20_match_wickets_2'].mean()
                    inference_bowler_details.loc[count, 'tboDa_dismissals_3'] =  ours['T20_match_wickets_3'].mean()
                    #inference_bowler_details.loc[count, 'tboDa_dismissals_agg'] =  ours['tboDa_dismissals_agg'].mean()
                    inference_bowler_details.loc[count, 'tboDa_bowledlbw_1'] =  ours['T20_match_bowledlbw_1'].mean()
                    inference_bowler_details.loc[count, 'tboDa_bowledlbw_2'] =  ours['T20_match_bowledlbw_2'].mean()
                    inference_bowler_details.loc[count, 'tboDa_bowledlbw_3'] =  ours['T20_match_bowledlbw_3'].mean()
                    #inference_bowler_details.loc[count, 'tboDa_bowledlbw_agg'] =  ours['tboDa_bowledlbw_agg'].mean()
                    inference_bowler_details.loc[count, 'tboDa_maidens_1'] =  ours['T20_match_maidens_1'].mean()
                    inference_bowler_details.loc[count, 'tboDa_maidens_2'] =  ours['T20_match_maidens_2'].mean()
                    inference_bowler_details.loc[count, 'tboDa_maidens_3'] =  ours['T20_match_maidens_3'].mean()
                    #inference_bowler_details.loc[count, 'tboDa_maidens_agg'] =  ours['tboDa_maidens_agg'].mean()
                
                ours = match_bowler_details[(match_bowler_details['player_name'] == player)].tail(10)
                if ours.empty:
                    inference_bowler_details.loc[count, 'fboDa_economy_1'] = 0
                    inference_bowler_details.loc[count, 'fboDa_economy_2'] = 0
                    inference_bowler_details.loc[count, 'fboDa_economy_3'] = 0
                    inference_bowler_details.loc[count, 'fboDa_economy_agg'] = 0
                    inference_bowler_details.loc[count, 'fboDa_dismissals_1'] = 0
                    inference_bowler_details.loc[count, 'fboDa_dismissals_2'] = 0
                    inference_bowler_details.loc[count, 'fboDa_dismissals_3'] = 0
                    inference_bowler_details.loc[count, 'fboDa_dismissals_agg'] = 0
                    inference_bowler_details.loc[count, 'fboDa_bowledlbw_1'] =  0
                    inference_bowler_details.loc[count, 'fboDa_bowledlbw_2'] =  0
                    inference_bowler_details.loc[count, 'fboDa_bowledlbw_3'] =  0
                    inference_bowler_details.loc[count, 'fboDa_bowledlbw_agg'] =  0
                    inference_bowler_details.loc[count, 'fboDa_maidens_1'] =  0
                    inference_bowler_details.loc[count, 'fboDa_maidens_2'] =  0
                    inference_bowler_details.loc[count, 'fboDa_maidens_3'] =  0
                    inference_bowler_details.loc[count, 'fboDa_maidens_agg'] =  0
                else:
                    inference_bowler_details.loc[count, 'fboDa_economy_1'] = sum(ours['T20_match_runs_1']) * 6 / (sum(ours['T20_match_balls_1']) if sum(ours['T20_match_balls_1']) != 0 else 1)
                    inference_bowler_details.loc[count, 'fboDa_economy_2'] = sum(ours['T20_match_runs_2']) * 6 / (sum(ours['T20_match_balls_2']) if sum(ours['T20_match_balls_2']) != 0 else 1)
                    inference_bowler_details.loc[count, 'fboDa_economy_3'] = sum(ours['T20_match_runs_3']) * 6 / (sum(ours['T20_match_balls_3']) if sum(ours['T20_match_balls_3']) != 0 else 1)
                    #inference_bowler_details.loc[count, 'fboDa_economy_agg'] = sum(ours['runs']) * 6 / (sum(ours['balls']) if sum(ours['balls']) != 0 else 1)
                    inference_bowler_details.loc[count, 'fboDa_dismissals_1'] =  ours['T20_match_wickets_1'].mean()
                    inference_bowler_details.loc[count, 'fboDa_dismissals_2'] =  ours['T20_match_wickets_2'].mean()
                    inference_bowler_details.loc[count, 'fboDa_dismissals_3'] =  ours['T20_match_wickets_3'].mean()
                    #inference_bowler_details.loc[count, 'fboDa_dismissals_agg'] =  ours['wickets'].mean()
                    inference_bowler_details.loc[count, 'fboDa_bowledlbw_1'] =  ours['T20_match_bowledlbw_1'].mean()
                    inference_bowler_details.loc[count, 'fboDa_bowledlbw_2'] =  ours['T20_match_bowledlbw_2'].mean()
                    inference_bowler_details.loc[count, 'fboDa_bowledlbw_3'] =  ours['T20_match_bowledlbw_3'].mean()
                    #inference_bowler_details.loc[count, 'fboDa_bowledlbw_agg'] =  ours['bowledlbw'].mean()
                    inference_bowler_details.loc[count, 'fboDa_maidens_1'] =  ours['T20_match_maidens_1'].mean()
                    inference_bowler_details.loc[count, 'fboDa_maidens_2'] =  ours['T20_match_maidens_2'].mean()
                    inference_bowler_details.loc[count, 'fboDa_maidens_3'] =  ours['T20_match_maidens_3'].mean()
                    #inference_bowler_details.loc[count, 'fboDa_maidens_agg'] =  ours['maidens'].mean()
                    #print(list(venue_dict_bowl[venue].keys()))
                    # venue_bowl_df.loc[len(venue_bowl_df)]=pd.DataFrame([venue_dict_bowl[venue]]).iloc[0]
                

      

            
    
             
       
                


          
                
          

                    
        
              
            
            
                player = str(player).replace(' (sub)', "")
                count=len(player_fielding)
                player_fielding.loc[count,'date'] = date
                player_fielding.loc[count,'player_name'] = player
                player_fielding.loc[count,'venue_name'] = venue
               
                try:
                    overall_fielding_details.index.get_loc(player)
                except:
                    
                    overall_fielding_details.loc[player,'caught'] = 0
                    overall_fielding_details.loc[player,'runouts'] = 0
                    overall_fielding_details.loc[player,'stumped'] = 0
                    overall_fielding_details.loc[player,'innings'] = 0
                
                player_fielding.loc[count,'previous_catches'] = overall_fielding_details.loc[player,'caught']
                player_fielding.loc[count,'previous_runouts'] = overall_fielding_details.loc[player,'runouts']
                player_fielding.loc[count,'previous_stumpings'] = overall_fielding_details.loc[player,'stumped']
                player_fielding.loc[count,'previous_innings_fielded'] = overall_fielding_details.loc[player,'innings']
            
                
                player_fielding.loc[count,'pFa_catches'] = (overall_fielding_details.loc[player,'caught'] / overall_fielding_details.loc[player,'innings']) if overall_fielding_details.loc[player,'innings'] != 0 else 0
                player_fielding.loc[count,'pFa_stumpings'] = (overall_fielding_details.loc[player,'stumped'] / overall_fielding_details.loc[player,'innings']) if overall_fielding_details.loc[player,'innings'] != 0 else 0
                player_fielding.loc[count,'pFa_runouts'] = (overall_fielding_details.loc[player,'runouts'] / overall_fielding_details.loc[player,'innings']) if overall_fielding_details.loc[player,'innings'] != 0 else 0
             
                # venue_field_df.loc[len(venue_field_df)]=pd.DataFrame([venue_dict_field[venue]]).iloc[0]
               
            checklist=[i[:-1] for i in list(venue_player_df) if '1' in i or '2' in i or '3' in i]
            checklist=list(pd.Series(checklist).unique())
            for j in checklist: 
                 
                if j[4]=='s':
                
                 venue_player_df['W'+j[:-1]]= (venue_player_df[j+'1']*inference_batsman_details['hist_spin_balls_b1']+venue_player_df[j+'2']*inference_batsman_details['hist_spin_balls_b2']+venue_player_df[j+'3']*inference_batsman_details['hist_spin_balls_b3'])/(inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details['hist_spin_balls_b2']+inference_batsman_details['hist_spin_balls_b3']+0.01)
                 venue_player_df['W'+j[:4]+j[5:-1]]= (venue_player_df[j+'1']*inference_batsman_details['hist_spin_balls_b1']+venue_player_df[j+'2']*inference_batsman_details['hist_spin_balls_b2']+venue_player_df[j+'3']*inference_batsman_details['hist_spin_balls_b3']+venue_player_df[j+'1']*inference_batsman_details['hist_pace_balls_b1']+venue_player_df[j+'2']*inference_batsman_details['hist_pace_balls_b2']+venue_player_df[j+'3']*inference_batsman_details['hist_pace_balls_b3'])/(inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details['hist_spin_balls_b2']+inference_batsman_details['hist_spin_balls_b3']+inference_batsman_details['hist_pace_balls_b1']+inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details['hist_pace_balls_b3']+0.01)

                if j[4]=='p':
                 venue_player_df['W'+j[:-1]]= (venue_player_df[j+'1']*inference_batsman_details['hist_pace_balls_b1']+venue_player_df[j+'2']*inference_batsman_details['hist_pace_balls_b2']+venue_player_df[j+'3']*inference_batsman_details['hist_pace_balls_b3'])/(inference_batsman_details['hist_pace_balls_b1']+inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details['hist_pace_balls_b3']+0.01)
            
                if j[4]=='r':
                 venue_player_df['W'+j[:-1]]= (venue_player_df[j+'1']*(inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details['hist_pace_balls_b1'])+venue_player_df[j+'2']*(inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details['hist_spin_balls_b2'])+venue_player_df[j+'3']*(inference_batsman_details['hist_spin_balls_b3']+inference_batsman_details['hist_pace_balls_b3']))/(inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details['hist_spin_balls_b2']+inference_batsman_details['hist_spin_balls_b3']+inference_batsman_details['hist_pace_balls_b1']+inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details['hist_pace_balls_b3']+0.01)
                
            checklist=[i[:-1] for i in list(inference_batsman_details) if '1' in i or '2' in i or '3' in i]
            checklist=list(pd.Series(checklist).unique())
            for j in checklist: 
                 
                if j[4]=='s':
                
                 inference_batsman_details['W'+j[:-1]]= (inference_batsman_details[j+'1']*inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details[j+'2']*inference_batsman_details['hist_spin_balls_b2']+inference_batsman_details[j+'3']*inference_batsman_details['hist_spin_balls_b3'])/(inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details['hist_spin_balls_b2']+inference_batsman_details['hist_spin_balls_b3']+0.01)
                 inference_batsman_details['W'+j[:4]+j[5:-1]]= (inference_batsman_details[j+'1']*inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details[j+'2']*inference_batsman_details['hist_spin_balls_b2']+inference_batsman_details[j+'3']*inference_batsman_details['hist_spin_balls_b3']+inference_batsman_details[j+'1']*inference_batsman_details['hist_pace_balls_b1']+inference_batsman_details[j+'2']*inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details[j+'3']*inference_batsman_details['hist_pace_balls_b3'])/(inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details['hist_spin_balls_b2']+inference_batsman_details['hist_spin_balls_b3']+inference_batsman_details['hist_pace_balls_b1']+inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details['hist_pace_balls_b3']+0.01)

                if j[4]=='p':
                 inference_batsman_details['W'+j[:-1]]= (inference_batsman_details[j+'1']*inference_batsman_details['hist_pace_balls_b1']+inference_batsman_details[j+'2']*inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details[j+'3']*inference_batsman_details['hist_pace_balls_b3'])/(inference_batsman_details['hist_pace_balls_b1']+inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details['hist_pace_balls_b3']+0.01)
            
                if j[4]=='r':
                 inference_batsman_details['W'+j[:-1]]= (inference_batsman_details[j+'1']*(inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details['hist_pace_balls_b1'])+inference_batsman_details[j+'2']*(inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details['hist_spin_balls_b2'])+inference_batsman_details[j+'3']*(inference_batsman_details['hist_spin_balls_b3']+inference_batsman_details['hist_pace_balls_b3']))/(inference_batsman_details['hist_spin_balls_b1']+inference_batsman_details['hist_spin_balls_b2']+inference_batsman_details['hist_spin_balls_b3']+inference_batsman_details['hist_pace_balls_b1']+inference_batsman_details['hist_pace_balls_b2']+inference_batsman_details['hist_pace_balls_b3']+0.01)
                
            inference_batsman_details.to_csv('data/inference/match_batsman_details.csv',index=False)
            venue_player_df.to_csv('data/inference/venue_player_df.csv',index=False)
            inference_bowler_details.to_csv('data/inference/match_bowler_details.csv',index=False)
            inference_bowler_details_venue.to_csv('data/inference/match_bowler_details_venue.csv',index=False)
            player_fielding.to_csv('data/inference/fielder.csv',index=False)
            
            playing_df.to_csv('data/inference/players.csv',index=False)
def matchups(player_df):
    matchup=[]
    for i in player_df['player_name']:
          for j in player_df['player_name']:
                if i!=j and  player_df[player_df['player_name']==i]['Team'].values[0]!=player_df[player_df['player_name']==j]['Team'].values[0]:
                    matchup.append((i,j))
    # print((matchup))
    final_matchups=pd.read_csv('data/matchups.csv')

    final_matchups['key']=[(final_matchups['batsman_name'][i],final_matchups['bowler_name'][i]) for i in range(len(final_matchups))]
    
    count=0
    matchups=pd.DataFrame(columns=[
        "date", "match_id", "batsman_name", "batsman_id", "bowler_name", "bowler_id",'batsman_team','bowler_team',
        "balls", "balls_involved", "runs", "strike_rate", "economy", "6s", "4s", "got_out", "maidens",
        "previous_innings_head_to_head", "previous_runs", "previous_4s", "previous_6s",
        "previous_avg_strike_rate", "previous_average", "previous_wickets", "previous_maidens",
        "previous_economy", 
        "h2h_boundary_percentage", "boundaries_per_dismissal", "clash_ratio",
        "batsman_dominance", "bowler_advantage", "h2h_consistency", "relative_strength_score","risk_reward"
    ])
    # print(final_matchups['key'])
    for key in matchup:
                matchups.loc[count, 'batsman_team']=player_df[player_df['player_name']==key[0]]['Team'].values[0]
                matchups.loc[count, 'bowler_team']=player_df[player_df['player_name']==key[1]]['Team'].values[0]
                matchups.loc[count, 'batsman_name']=key[0]
                matchups.loc[count, 'bowler_name']=key[1]
                count+=1
                # print('ligga',final_matchups[final_matchups['key']==key])#['previous_wickets']+final_matchups[final_matchups['key']==key]['T20_match_wickets']
                if key in dict1['total_runs'].keys(): 
                    matchups.loc[count, 'previous_runs'] =  dict1['total_runs'][key]
                    matchups.loc[count, 'previous_4s'] = dict1['total_4s'][key] 
                    matchups.loc[count, 'previous_6s'] = dict1['total_6s'][key]
                   
                    if  dict1['total_balls'][key] == 0:
                        matchups.loc[count, 'previous_avg_strike_rate'] =   dict1['total_runs'][key]
                    else:
                        matchups.loc[count, 'previous_avg_strike_rate'] =  dict1['total_runs'][key]/(dict1['total_balls'][key]+0.1)#(final_matchups[final_matchups['key']==key]['previous_runs']+final_matchups[final_matchups['key']==key]['T20_match_runs'])* 100 /(final_matchups[final_matchups['key']==key]['T20_match_balls'].sum() )
                    try:
                        matchups.loc[count, 'previous_average'] =  dict1['total_runss'][key]/(dict1['total_wickets'][key]+0.1)#(final_matchups[final_matchups['key']==key]['previous_runs']+final_matchups[final_matchups['key']==key]['T20_match_runs'])/ (final_matchups[final_matchups['key']==key]['previous_wickets'])
                    except:
                        # print('nigga')
                        matchups.loc[count, 'previous_average'] = dict1['total_runs'][key]
                    matchups.loc[count, 'previous_innings_head_to_head'] =  dict1['total_innings'][key]
                    # matchups.loc[count, 'previous_maidens'] =  final_matchups[final_matchups['key']==key]['T20_match_maidens']
                    matchups.loc[count, 'previous_economy'] = (dict1['total_runs'][key] * 6 / (dict1['total_balls'][key] +0.1))
                    matchups.loc[count,"previous_maidens"]=dict1['total_maidens'][key]
                     
    
                else:
                 matchups.loc[count,"previous_maidens"]=0.1
                 matchups.loc[count, 'previous_runs'] =0.1
                 matchups.loc[count, 'previous_4s'] =0.1
                 matchups.loc[count, 'previous_6s'] =0.1
                 matchups.loc[count,"prev_5_avg_sr"] =0.1
                 matchups.loc[count, 'previous_average'] = 0.1
                 matchups.loc[count, 'previous_avg_strike_rate'] =0.1# (final_matchups[final_matchups['key']==key]['previous_runs']+final_matchups[final_matchups['key']==key]['T20_match_runs'])
                 matchups.loc[count, 'previous_innings_head_to_head'] = 0.1# sum([final_matchups['key']==key])
                    # matchups.loc[count, 'previous_maidens'] =  final_matchups[final_matchups['key']==key]['T20_match_maidens']
                 matchups.loc[count, 'previous_economy'] = 0.1
                try: 
                     matchups.loc[count,"prev_5_avg_sr"] = final_matchups[final_matchups['key']==key]['T20_match_strike_rate'][-5:].mean()#.fillna(0)
                    #  print(final_matchups[final_matchups['key']==key]['T20_match_strike_rate'][-5:])
                     
                     if final_matchups[final_matchups['key']==key]['T20_match_strike_rate'][-5:].empty==False:
                          print(final_matchups[final_matchups['key']==key]['T20_match_strike_rate'][-5:].mean(),'mean')
                          print(matchups.loc[count,"prev_5_avg_sr"])
                except:
                     print('nig')
                     matchups.loc[count,"prev_5_avg_sr"] =0.1#( (final_matchups[final_matchups['key']==key]['previous_runs']+final_matchups[final_matchups['key']==key]['T20_match_runs']) * 6 /  final_matchups[final_matchups['key']==key]['T20_match_balls'].sum()) if  final_matchups[final_matchups['key']==key]['T20_match_balls'].sum() != 0 else 0
    matchups["total_h2h_boundaries"] = matchups["previous_4s"] + matchups["previous_6s"]
    matchups=matchups.loc[:241]

    # Calculate balls faced in historical H2H (derived from strike rate)
    matchups["previous_balls"] = ((matchups["previous_runs"] )/ (matchups["previous_avg_strike_rate"]+0.1) * 100).replace(np.inf, 0).fillna(0)

    # Boundary Percentage
    matchups["h2h_boundary_percentage"] = np.where(
        matchups["previous_balls"] > 0,
        (matchups["total_h2h_boundaries"] / (matchups["previous_balls"]+0.001)) * 100,
        0  # Default for no historical balls
    )

    # Boundaries per dismissal (risk vs reward)
    matchups["boundaries_per_dismissal"] = np.where(
        matchups["previous_wickets"] > 0,
        (matchups["total_h2h_boundaries"] / matchups["previous_wickets"]),
        matchups["total_h2h_boundaries"]  # If no dismissals, use total boundaries as-is
    )

    # ----------------------------------------------------
    # 2. Pressure Response (Requires Innings Data - Example Implementation)
    # ----------------------------------------------------
    # Assuming you have a column "innings" (1 or 2) in your CSV
    # matchups["pressure_response"] = matchups.groupby(["batsman_id", "bowler_id", "innings"])["strike_rate"].transform(
    #     lambda x: x[x.innings == 2].mean() / x[x.innings == 1].mean() if len(x) > 1 else 1
    # )

    # ----------------------------------------------------
    # 3. Clash Ratio (Team-Level Dynamics)
    # ----------------------------------------------------
    # Batsman team's historical average runs (at venue if available)
    batsman_team_avg = matchups.groupby("batsman_team")["previous_runs"].transform("mean")

    # Bowler team's historical economy (at venue if available)
    bowler_team_avg_eco = matchups.groupby("bowler_team")["previous_economy"].transform("mean")

    matchups["clash_ratio"] = batsman_team_avg / bowler_team_avg_eco.replace(0, 1)  # Avoid division by zero



    # Calculate rolling average of strike rate for last 5 matches (excluding current row)
    # matchups["prev_5_avg_sr"] = (
    #     matchups.groupby("batsman_id")["T20_Match_strike_rate"]
    #     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    # )

    # Handle NaNs (e.g., for batsmen with <5 matches)
    
    


    matchups["prev_5_eco"] = (
        matchups.groupby("bowler_id")["economy"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Handle NaNs (e.g., for batsmen with <5 matches)
    matchups["prev_5_eco"] = matchups.groupby("bowler_id")["prev_5_eco"].transform(
        lambda x: x.fillna(x.mean())
    )

    # ----------------------------------------------------
    # 4. Dominance Metrics
    # ----------------------------------------------------
    # Batsman Dominance

    matchups["batsman_dominance"] = (matchups["prev_5_avg_sr"] / matchups["previous_avg_strike_rate"].replace(0, 1)) 

    # Bowler Advantage
    matchups["bowler_advantage"] = (matchups["prev_5_eco"] / matchups["previous_economy"].replace(0, 1)) * 100  # Avoid division by zero

    # ----------------------------------------------------
    # 5. Consistency Metric
    # ----------------------------------------------------
    # (Assuming you have match-by-match historical runs in a long format)
    # For simplicity, we'll approximate using historical data variability
    matchups["h2h_consistency"] = matchups["previous_avg_strike_rate"] /( matchups["previous_avg_strike_rate"].rolling(5, min_periods=1).std()+0.01)

    # ----------------------------------------------------
    # 6. Relative Strength Score (Composite Metric)
    # ----------------------------------------------------
    # Normalize components
    features_to_combine = [
        "h2h_boundary_percentage", 
        "batsman_dominance",
        "h2h_consistency",
        "clash_ratio",
        "bowler_advantage"
    ]

    # Z-score normalization
    matchups_normalized = (matchups[features_to_combine] - matchups[features_to_combine].mean()) / (matchups[features_to_combine].std()+0.01)

    # Weighted combination (adjust weights as needed)
    weights = [0.3, 0.2, 0.3, 0.1,0.2]  
    matchups["relative_strength_score"] = matchups_normalized.dot(weights)
    matchups["risk_reward"] = (matchups["boundaries_per_dismissal"] * 0.6) + (matchups["prev_5_avg_sr"] * 0.4)


        
        
        
    # ----------------------------------------------------
    # Drop Unnecessary Columns
    # ----------------------------------------------------
    # List of columns to keep in the final output
    final_columns = [
        "date",  "batsman_name", "bowler_name",
       
        "previous_innings_head_to_head", "previous_runs", "previous_4s", "previous_6s",
        "previous_avg_strike_rate", "previous_average", "previous_wickets", "previous_maidens",
        "previous_economy", "batsman_team", "bowler_team","prev_5_avg_sr",
        "h2h_boundary_percentage", "boundaries_per_dismissal", "clash_ratio",
        "batsman_dominance", "bowler_advantage", "h2h_consistency", "relative_strength_score","risk_reward"
    ]

    # Drop unnecessary columns
    matchups = matchups[final_columns]

    # ----------------------------------------------------
    # Save Enhanced Dataset
    # ----------------------------------------------------
    matchups.to_csv("data/inference/enhanced_matchups.csv", index=False) 
def generate_batsman_vs_opposition_metrics(h2h_csv,h2h_hist, batter_csv, bowler_info_csv,match_date):
    # Load CSVs
    h2h_data = pd.read_csv(h2h_csv)
    h2h_hist_data=pd.read_csv(h2h_hist)
    playing_xi = pd.read_csv(batter_csv)
    bowler_info = pd.read_csv(bowler_info_csv)

    # Filter only playing 11
    #playing_xi = playing_xi[playing_xi['IsPlaying'] == 'PLAYING']

    #print(playing_xi['Team'].unique())  # Should be only ['India', 'South Africa']
    #print(playing_xi['player_name'].unique())  # Check for any unusual names

    # Merge bowler info to get bowler types
    h2h_data = h2h_data.merge(bowler_info, left_on='bowler_name', right_on='player_name', how='left')
    h2h_hist_data = h2h_hist_data.merge(bowler_info, left_on='bowler_name', right_on='player_name', how='left')
    left_arm_styles = {'LPACE', 'LLEG', 'LOFF'}
    right_arm_styles = {'RPACE', 'RLEG', 'ROFF'}

    metrics = []


    def calculate_stats(df,df2):
        total_runs = df['previous_runs'].sum()
        total_balls = df2['balls_involved'].sum()
        strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0
        boundary_percentage = ((df['previous_4s'].sum() + df['previous_6s'].sum()) / total_balls) * 100 if total_balls > 0 else 0
        dismissal_rate = total_balls / df['previous_wickets'].sum() if df['previous_wickets'].sum() > 0 else 0
        h2h_avg = total_runs/ df['previous_wickets'].sum() if df['previous_wickets'].sum() > 0 else 0
        impact_index = (strike_rate * boundary_percentage) / 100
        
        return total_runs, total_balls, strike_rate, boundary_percentage, dismissal_rate, h2h_avg, impact_index

    
    for batsman in playing_xi['player_name']:
        batsman_team = playing_xi.loc[playing_xi['player_name'] == batsman, 'team']

        if batsman_team.empty:
            continue  # Skip this batsman if not found in playing_xi

        batsman_team = batsman_team.iloc[0]  # Extract the single team value

        opposition_bowlers = playing_xi[playing_xi['opposition_name'] == batsman_team]['player_name'].tolist()

        batsman_data = h2h_data[(h2h_data['batsman_name'] == batsman)]
        batsman_vs_opposition = batsman_data[batsman_data['bowler_name'].isin(opposition_bowlers)]
        print(batsman_vs_opposition['bowling_style'])
        batsman_data2 = h2h_hist_data[(h2h_hist_data['batsman_name'] == batsman)]
        batsman_vs_opposition2 = batsman_data2[batsman_data2['bowler_name'].isin(opposition_bowlers)]


        # Segregate bowlers into Spinners and Pacers
        batsman_vs_spinners = batsman_vs_opposition[batsman_vs_opposition['bowling_category'] == 'SPIN']
        batsman_vs_pacers = batsman_vs_opposition[batsman_vs_opposition['bowling_category'] == 'PACE']

        batsman_vs_spinners2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_category'] == 'SPIN']
        batsman_vs_pacers2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_category'] == 'PACE']

        batsman_vs_left_arm = batsman_vs_opposition[batsman_vs_opposition['bowling_style.1'].isin(left_arm_styles)]
        batsman_vs_right_arm = batsman_vs_opposition[batsman_vs_opposition['bowling_style.1'].isin(right_arm_styles)]
        batsman_vs_lleg = batsman_vs_opposition[batsman_vs_opposition['bowling_style.1'] == 'LLEG']
        batsman_vs_rleg = batsman_vs_opposition[batsman_vs_opposition['bowling_style.1'] == 'RLEG']
        batsman_vs_loff = batsman_vs_opposition[batsman_vs_opposition['bowling_style.1'] == 'LOFF']
        batsman_vs_roff = batsman_vs_opposition[batsman_vs_opposition['bowling_style.1'] == 'ROFF']
        batsman_vs_lpace = batsman_vs_opposition[batsman_vs_opposition['bowling_style.1'] == 'LPACE']
        batsman_vs_rpace = batsman_vs_opposition[batsman_vs_opposition['bowling_style.1'] == 'RPACE']

        batsman_vs_left_arm2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_style.1'].isin(left_arm_styles)]
        batsman_vs_right_arm2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_style.1'].isin(right_arm_styles)]
        batsman_vs_lleg2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_style.1'] == 'LLEG']
        batsman_vs_rleg2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_style.1'] == 'RLEG']
        batsman_vs_loff2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_style.1'] == 'LOFF']
        batsman_vs_roff2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_style.1'] == 'ROFF']
        batsman_vs_lpace2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_style.1'] == 'LPACE']
        batsman_vs_rpace2 = batsman_vs_opposition2[batsman_vs_opposition2['bowling_style.1'] == 'RPACE']

        # Aggregate Stats
        total_runs = batsman_vs_opposition['previous_runs'].sum()
        total_balls = batsman_vs_opposition2['balls_involved'].sum()
        total_4s = batsman_vs_opposition['previous_4s'].sum()
        total_6s = batsman_vs_opposition['previous_6s'].sum()
        total_outs = batsman_vs_opposition['previous_wickets'].sum()
        #dot_balls = batsman_vs_opposition['balls_involved'].sum() - (total_4s + total_6s + total_runs)
        maidens_faced = batsman_vs_opposition['previous_maidens'].sum()

        # Derived Metrics
        strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0
        boundary_percentage = ((total_4s + total_6s) / total_balls) * 100 if total_balls > 0 else 0
        balls_per_boundary = total_balls / (total_4s + total_6s) if (total_4s + total_6s) > 0 else 0
        #dot_ball_percentage = (dot_balls / total_balls) * 100 if total_balls > 0 else 0
        balls_per_run = total_balls / total_runs if total_runs > 0 else 0
        runs_per_maiden = total_runs / maidens_faced if maidens_faced > 0 else 0
        dismissal_rate = total_balls / total_outs if total_outs > 0 else 0
        risk_factor = total_6s / total_outs if total_outs > 0 else 0

        # Pressure & Situational Metrics
        #pressure_response = batsman_vs_opposition['previous_avg_strike_rate'].mean() / batsman_vs_opposition['previous_avg_strike_rate'].mean()

        # Head-to-Head Performance
        head_to_head_avg = total_runs / total_outs if total_outs > 0 else 0
        bowler_exploitability = batsman_vs_opposition['previous_runs'].sum() / batsman_vs_opposition['previous_economy'].mean()

        # Dominance & Strength Metrics
        #batsman_dominance = (head_to_head_avg / batsman_vs_opposition['previous_average'].mean()) * 100 if batsman_vs_opposition['previous_average'].mean() > 0 else 0
        #bowler_advantage = (batsman_vs_opposition['previous_economy'].mean() / batsman_vs_opposition['economy'].mean()) * 100 if batsman_vs_opposition['economy'].mean() > 0 else 0

        # Consistency & Adaptability
        h2h_consistency = head_to_head_avg / batsman_vs_opposition['previous_runs'].std() if batsman_vs_opposition['previous_runs'].std() > 0 else 0
        adaptability_score = (strike_rate + h2h_consistency) / 2

        # Overall Impact Score
        #impact_index = (strike_rate * boundary_percentage * pressure_response) / 100

        # Calculate Metrics for Spinners
        total_runs_vs_spinners = batsman_vs_spinners['previous_runs'].sum()
        total_balls_vs_spinners = batsman_vs_spinners2['balls_involved'].sum()
        strike_rate_vs_spinners = (total_runs_vs_spinners / total_balls_vs_spinners) * 100 if total_balls_vs_spinners > 0 else 0
        boundary_percentage_vs_spinners = ((batsman_vs_spinners['previous_4s'].sum() + batsman_vs_spinners['previous_6s'].sum()) / total_balls_vs_spinners) * 100 if total_balls_vs_spinners > 0 else 0
        dismissal_rate_vs_spinners = total_balls_vs_spinners / batsman_vs_spinners['previous_wickets'].sum() if batsman_vs_spinners['previous_wickets'].sum() > 0 else 0
        h2h_avg_vs_spinners = total_runs_vs_spinners / batsman_vs_spinners['previous_wickets'].sum() if batsman_vs_spinners['previous_wickets'].sum() > 0 else 0
        impact_index_vs_spinners = (strike_rate_vs_spinners * boundary_percentage_vs_spinners) / 100
        batsman_dominance_vs_spinners = (h2h_avg_vs_spinners / batsman_vs_spinners['previous_average'].mean()) * 100 if batsman_vs_spinners['previous_average'].mean() > 0 else 0
        #spinners_advantage_vs_batsman = (batsman_vs_spinners['previous_economy'].mean() / batsman_vs_spinners['economy'].mean()) * 100 if batsman_vs_spinners['economy'].mean() > 0 else 0

        # Calculate Metrics for Pacers
        total_runs_vs_pacers = batsman_vs_pacers['previous_runs'].sum()
        total_balls_vs_pacers = batsman_vs_pacers2['balls_involved'].sum()
        strike_rate_vs_pacers = (total_runs_vs_pacers / total_balls_vs_pacers) * 100 if total_balls_vs_pacers > 0 else 0
        boundary_percentage_vs_pacers = ((batsman_vs_pacers['previous_4s'].sum() + batsman_vs_pacers['previous_6s'].sum()) / total_balls_vs_pacers) * 100 if total_balls_vs_pacers > 0 else 0
        dismissal_rate_vs_pacers = total_balls_vs_pacers / batsman_vs_pacers['previous_wickets'].sum() if batsman_vs_pacers['previous_wickets'].sum() > 0 else 0
        h2h_avg_vs_pacers = total_runs_vs_pacers / batsman_vs_pacers['previous_wickets'].sum() if batsman_vs_pacers['previous_wickets'].sum() > 0 else 0
        impact_index_vs_pacers = (strike_rate_vs_pacers * boundary_percentage_vs_pacers) / 100
        batsman_dominance_vs_pacers = (h2h_avg_vs_pacers / batsman_vs_pacers['previous_average'].mean()) * 100 if batsman_vs_pacers['previous_average'].mean() > 0 else 0
        #pacers_advantage_vs_batsman = (batsman_vs_pacers['previous_economy'].mean() / batsman_vs_pacers['economy'].mean()) * 100 if batsman_vs_pacers['economy'].mean() > 0 else 0

     

        # Store Metrics
        metrics.append({
           
            'batsman': batsman,
            'total_runs_vs_opposition': total_runs,
            #'total_balls_vs_opposition': total_balls,
            'strike_rate': strike_rate,
            'boundary_percentage': boundary_percentage,
            'balls_per_boundary': balls_per_boundary,
            #'dot_ball_percentage': dot_ball_percentage,
            'balls_per_run': balls_per_run,
            'runs_per_maiden': runs_per_maiden,
            'dismissal_rate': dismissal_rate,
            'risk_factor': risk_factor,
            #'pressure_response': pressure_response,
            'head_to_head_avg': head_to_head_avg,
            'bowler_exploitability': bowler_exploitability,
            #'batsman_dominance': batsman_dominance,
            #'bowler_advantage': bowler_advantage,
            'h2h_consistency': h2h_consistency,
            'adaptability_score': adaptability_score,
            #'impact_index': impact_index,
            #'relative_strength_score': (batsman_dominance + h2h_consistency + adaptability_score) / 3,

            'total_runs_vs_spinners': total_runs_vs_spinners,
            #'total_balls_vs_spinners': total_balls_vs_spinners,
            'strike_rate_vs_spinners': strike_rate_vs_spinners,
            'boundary_percentage_vs_spinners': boundary_percentage_vs_spinners,
            'dismissal_rate_vs_spinners': dismissal_rate_vs_spinners,
            'h2h_avg_vs_spinners': h2h_avg_vs_spinners,
            'impact_index_vs_spinners': impact_index_vs_spinners,
            'batsman_dominance_vs_spinners': batsman_dominance_vs_spinners,
            #'spinners_advantage_vs_batsman': spinners_advantage_vs_batsman,

            'total_runs_vs_pacers': total_runs_vs_pacers,
            #'total_balls_vs_pacers': total_balls_vs_pacers,
            'strike_rate_vs_pacers': strike_rate_vs_pacers,
            'boundary_percentage_vs_pacers': boundary_percentage_vs_pacers,
            'dismissal_rate_vs_pacers': dismissal_rate_vs_pacers,
            'h2h_avg_vs_pacers': h2h_avg_vs_pacers,
            'impact_index_vs_pacers': impact_index_vs_pacers,
            'batsman_dominance_vs_pacers': batsman_dominance_vs_pacers,
            #'pacers_advantage_vs_batsman': pacers_advantage_vs_batsman,

            'total_runs_vs_left_arm': calculate_stats(batsman_vs_left_arm,batsman_vs_left_arm2 )[0],
            #'total_balls_vs_left_arm': calculate_stats(batsman_vs_left_arm)[1],
            'strike_rate_vs_left_arm': calculate_stats(batsman_vs_left_arm,batsman_vs_left_arm2)[2],
            'boundary_percentage_vs_left_arm': calculate_stats(batsman_vs_left_arm,batsman_vs_left_arm2)[3],
            'dismissal_rate_vs_left_arm': calculate_stats(batsman_vs_left_arm,batsman_vs_left_arm2)[4],
            'h2h_avg_vs_left_arm': calculate_stats(batsman_vs_left_arm,batsman_vs_left_arm2)[5],
            'impact_index_vs_left_arm': calculate_stats(batsman_vs_left_arm,batsman_vs_left_arm2)[6],

            'total_runs_vs_right_arm': calculate_stats(batsman_vs_right_arm,batsman_vs_right_arm2)[0],
            #'total_balls_vs_right_arm': calculate_stats(batsman_vs_right_arm)[1],
            'strike_rate_vs_right_arm': calculate_stats(batsman_vs_right_arm,batsman_vs_right_arm2)[2],
            'boundary_percentage_vs_right_arm': calculate_stats(batsman_vs_right_arm,batsman_vs_right_arm2)[3],
            'dismissal_rate_vs_right_arm': calculate_stats(batsman_vs_right_arm,batsman_vs_right_arm2)[4],
            'h2h_avg_vs_right_arm': calculate_stats(batsman_vs_right_arm,batsman_vs_right_arm2)[5],
            'impact_index_vs_right_arm': calculate_stats(batsman_vs_right_arm,batsman_vs_right_arm2)[6],

            'total_runs_vs_rleg': calculate_stats(batsman_vs_rleg,batsman_vs_rleg2)[0],
            'strike_rate_vs_rleg': calculate_stats(batsman_vs_rleg,batsman_vs_rleg2)[2],
            'boundary_percentage_vs_rleg': calculate_stats(batsman_vs_rleg,batsman_vs_rleg2)[3],
            'dismissal_rate_vs_rleg': calculate_stats(batsman_vs_rleg,batsman_vs_rleg2)[4],
            'h2h_avg_vs_rleg': calculate_stats(batsman_vs_rleg,batsman_vs_rleg2)[5],
            'impact_index_vs_rleg': calculate_stats(batsman_vs_rleg,batsman_vs_rleg2)[6],

            'total_runs_vs_lleg': calculate_stats(batsman_vs_lleg,batsman_vs_lleg2)[0],
            'strike_rate_vs_lleg': calculate_stats(batsman_vs_lleg,batsman_vs_lleg2)[2],
            'boundary_percentage_vs_lleg': calculate_stats(batsman_vs_lleg,batsman_vs_lleg2)[3],
            'dismissal_rate_vs_lleg': calculate_stats(batsman_vs_lleg,batsman_vs_lleg2)[4],
            'h2h_avg_vs_lleg': calculate_stats(batsman_vs_lleg,batsman_vs_lleg2)[5],
            'impact_index_vs_lleg': calculate_stats(batsman_vs_lleg,batsman_vs_lleg2)[6],

            'total_runs_vs_loff': calculate_stats(batsman_vs_loff,batsman_vs_loff2)[0],
            'strike_rate_vs_loff': calculate_stats(batsman_vs_loff,batsman_vs_loff2)[2],
            'boundary_percentage_vs_loff': calculate_stats(batsman_vs_loff,batsman_vs_loff2)[3],
            'dismissal_rate_vs_loff': calculate_stats(batsman_vs_loff,batsman_vs_loff2)[4],
            'h2h_avg_vs_loff': calculate_stats(batsman_vs_loff,batsman_vs_loff2)[5],
            'impact_index_vs_loff': calculate_stats(batsman_vs_loff,batsman_vs_loff2)[6],

            'total_runs_vs_roff': calculate_stats(batsman_vs_roff,batsman_vs_roff2)[0],
            'strike_rate_vs_roff': calculate_stats(batsman_vs_roff,batsman_vs_roff2)[2],
            'boundary_percentage_vs_roff': calculate_stats(batsman_vs_roff,batsman_vs_roff2)[3],
            'dismissal_rate_vs_roff': calculate_stats(batsman_vs_roff,batsman_vs_roff2)[4],
            'h2h_avg_vs_roff': calculate_stats(batsman_vs_roff,batsman_vs_roff2)[5],
            'impact_index_vs_roff': calculate_stats(batsman_vs_roff,batsman_vs_roff2)[6],

            'total_runs_vs_lpace': calculate_stats(batsman_vs_lpace,batsman_vs_lpace2)[0],
            'strike_rate_vs_lpace': calculate_stats(batsman_vs_lpace,batsman_vs_lpace2)[2],
            'boundary_percentage_vs_lpace': calculate_stats(batsman_vs_lpace,batsman_vs_lpace2)[3],
            'dismissal_rate_vs_lpace': calculate_stats(batsman_vs_lpace,batsman_vs_lpace2)[4],
            'h2h_avg_vs_lpace': calculate_stats(batsman_vs_lpace,batsman_vs_lpace2)[5],
            'impact_index_vs_lpace': calculate_stats(batsman_vs_lpace,batsman_vs_lpace2)[6],

            'total_runs_vs_rpace': calculate_stats(batsman_vs_rpace,batsman_vs_rpace2)[0],
            'strike_rate_vs_rpace': calculate_stats(batsman_vs_rpace,batsman_vs_rpace2)[2],
            'boundary_percentage_vs_rpace': calculate_stats(batsman_vs_rpace,batsman_vs_rpace2)[3],
            'dismissal_rate_vs_rpace': calculate_stats(batsman_vs_rpace,batsman_vs_rpace2)[4],
            'h2h_avg_vs_rpace': calculate_stats(batsman_vs_rpace,batsman_vs_rpace2)[5],
            'impact_index_vs_rpace': calculate_stats(batsman_vs_rpace,batsman_vs_rpace2)[6]
        })

    return pd.DataFrame(metrics)
def calculate_batsman_points_T20(runs, boundaries, sixes, strike_rate, balls_faced):
    batsman_points = 0

    # Run points
    batsman_points += runs

    # Boundary points
    batsman_points += boundaries * 1  # 1 point per boundary

    # Six points
    batsman_points += sixes * 2  # 2 points per six

    # Half-century & Century points
    batsman_points += np.where(runs>100, 8, 0)  # 100 runs = 8 points
    batsman_points += np.where(runs>50, 4, 0)  # 50 runs = 4 points

    # Strike rate bonus points (if player faced at least 20 balls)
    # Avoid division by zero by checking balls_faced first
    if balls_faced > 0:
        runs_per_100_balls = (runs / balls_faced) * 100
    else:
        runs_per_100_balls = 0  # Default to 0 if no balls faced

    # Apply strike rate bonuses
    batsman_points += np.where((balls_faced >= 20) & (runs_per_100_balls > 140), 6, 0)  # Above 140 runs per 100 balls
    batsman_points += np.where((balls_faced >= 10) & (runs_per_100_balls <= 140) & (runs_per_100_balls>120), 4, 0)  # Between 120.01 and 140 runs per 100 balls
    batsman_points += np.where((balls_faced >= 10) & (runs_per_100_balls <= 120) & (runs_per_100_balls>100), 2, 0)  # Between 100 and 120 runs per 100 balls
    batsman_points += np.where((balls_faced >= 10) & (runs_per_100_balls <= 50) & (runs_per_100_balls>=40), -2, 0)  # Between 40 and 50 runs per 100 balls
    batsman_points += np.where((balls_faced >= 10) & (runs_per_100_balls < 40) & (runs_per_100_balls>=30), -4, 0)   # Between 30 and 39.99 runs per 100 balls
    batsman_points += np.where((balls_faced >= 20) & (runs_per_100_balls < 30), -6, 0)  # Below 30 runs per 100 balls
    
    return batsman_points


def calculate_bowler_points_T20(wickets, bowled_lbw, maidens, economy, balls):
    bowler_points = 0

    # Wicket points (excluding run outs)
    bowler_points += wickets * 30  # 25 points per wicket

    # Bowled/LBW bonus points (8 points per wicket for Bowled or LBW)
    bowler_points += bowled_lbw * 8  # Each Bowled/LBW wicket adds 8 points

    # Wicket bonus points
    bowler_points += np.where((wickets >= 5), 8, 0)  # 5 wickets = 8 points
    bowler_points += np.where((wickets == 4), 4, 0)  # 4 wickets = 4 points

    # Maiden Over points
    bowler_points += maidens * 4  # 4 points per maiden over

    # Economy points (only if bowled at least 12 balls)
    if balls > 0:  # Avoid division by zero
        bowler_points += np.where((balls >= 30) & (economy < 2.5), 6, 0)   # Below 2.5 runs per over
        bowler_points += np.where((balls >= 12) & (economy>=2.5) & (economy<3.5), 4, 0)  # Between 2.5 and 3.49 runs per over
        bowler_points += np.where((balls >= 12) & (economy>=3.5) & (economy<=4.5), 2, 0)  # Between 3.5 and 4.5 runs per over
        bowler_points += np.where((balls >= 12) & (economy>=7) & (economy<=8), -2, 0)  # Between 7 and 8 runs per over
        bowler_points += np.where((balls >= 12) & (economy>8) & (economy<=9), -4, 0)  # Between 8.01 and 9 runs per over
        bowler_points += np.where((balls >= 30) & (economy >9), -6, 0)   # Above 9 runs per over

    return bowler_points


def calculate_fielder_points_T20(catches, stumpings):
    fielder_points = 0

    # Catches: +8 points for each catch
    fielder_points += catches * 8

    # 3 Catch Bonus: +4 points if 3 or more catches are taken
    # if catches >= 3:
    fielder_points  +=np.where(catches>=3,4,0)


    # Stumpings: +12 points for each stumping
    fielder_points += stumpings * 12

    # fielder_points += runouts * 9

    return fielder_points


def points_calculator(format, runs, boundaries, sixes, strike_rate, wickets, bowled_lbw, maidens, economy, catches, stumpings, balls_bowled=30):
    # Handle division by zero in strike rate calculation
    if strike_rate == 0 or np.isnan(strike_rate):
        balls_faced_while_batting = 0
    else:
        try:
            balls_faced_while_batting = runs*100/strike_rate
            if np.isinf(balls_faced_while_batting) or np.isnan(balls_faced_while_batting):
                balls_faced_while_batting = 0
        except:
            balls_faced_while_batting = 0

    format = format.lower()

    # Handle potential None or NaN values
  

    total_points = (calculate_batsman_points_T20(runs, boundaries, sixes, strike_rate, balls_faced_while_batting) +
                   calculate_bowler_points_T20(wickets, bowled_lbw, maidens, economy, balls_bowled) +
                   calculate_fielder_points_T20(catches, stumpings))
    
    return total_points
def process_predictions(predictions_df, input_df):
    print("\nProcessing predictions for input players:")
    print("Input players:", input_df['player_name'].values)
    # Save raw predictions for debugging/analysis
    # predictions_df.to_csv('app/raw_predictions.csv', index=False)    
    # Filter predictions only for input players
    predictions_df = predictions_df[predictions_df['player_name'].isin(input_df['player_name'])]
    
    # Group predictions by player name and pivot
    final_predictions = predictions_df
    
    # Merge with input dataframe
    final_predictions = pd.merge(
        input_df[['player_name', 'Team', 'Credits', 'Player Type', 'IsPlaying', 'lineupOrder']], 
        final_predictions,
        on='player_name',
        how='left'
    )
    print(final_predictions)
    # Create player type indicator columns
    final_predictions['batter'] = (final_predictions['Player Type_x'] == 'BAT').astype(int)
    final_predictions['bowler'] = (final_predictions['Player Type_x'] == 'BOWL').astype(int)
    final_predictions['all'] = (final_predictions['Player Type_x'] == 'ALL').astype(int)
    final_predictions['wk'] = (final_predictions['Player Type_x'] == 'WK').astype(int)
    
    # Create team indicators
    teams = final_predictions['Team_x'].unique()
    final_predictions['team_indicators'] = (final_predictions['Team_x'] == teams[0]).astype(int)
    
    # Calculate scores
    final_predictions['scores'] = final_predictions.apply(lambda row: points_calculator(
        format='T20',
        runs=row.get('T20_match_runs', 0),
        boundaries=row.get('T20_match_4s', 0),
        sixes=row.get('T20_match_6s', 0),
        strike_rate=row.get('T20_match_strike_rate', 0),
        wickets=row.get('T20_match_wickets', 0),
        bowled_lbw=row.get('T20_match_bowledlbw', 0),
        maidens=row.get('T20_match_maidens', 0),
        economy=row.get('T20_match_economy', 0),
        catches=row.get('match_catches', 0),
        stumpings=row.get('match_stumpings', 0)
    ), axis=1)
    
    # Save the DataFrame that's passed to optimizer
    # final_predictions.to_csv('app/optimizer_input.csv', index=False)
    
    print("\nProcessed predictions:")
    print("Number of players with predictions:", len(final_predictions))
    
    return final_predictions



def optimize_team(df, K=100):
    # Create the model
    prob = LpProblem("Dream11_Team_Selection", LpMaximize)
    
    # Add solver path (update path if cbc.exe is located elsewhere)
    solver = COIN_CMD(path="cbc.exe")  # Provide the correct path to cbc.exe
    
    n_players = len(df)
    x = LpVariable.dicts("select", range(n_players), cat='Binary')
    
    # Objective: Maximize total predicted points
    prob += lpSum([x[i] * df['scores'].iloc[i] for i in range(n_players)])
    
    # Constraints
    # Total players = 11
    prob += lpSum([x[i] for i in range(n_players)]) == 11
    
    # Position constraints
    prob += lpSum([x[i] * df['batter'].iloc[i] for i in range(n_players)]) >= 3  # Min 3 batsmen
    prob += lpSum([x[i] * df['bowler'].iloc[i] for i in range(n_players)]) >= 1 
    prob += lpSum([x[i] * df['bowler'].iloc[i] for i in range(n_players)]) <= 3# Min 3 bowlers
    prob += lpSum([x[i] * df['wk'].iloc[i] for i in range(n_players)]) >= 1  # 1 wicket-keeper
    prob += lpSum([x[i] * df['all'].iloc[i] for i in range(n_players)]) >= 1  # Min 1 all-rounder
    
    # Team constraints (max 7 from one team)
    team_0_players = lpSum([x[i] * (1-df['team_indicators'].iloc[i]) for i in range(n_players)])
    team_1_players = lpSum([x[i] * df['team_indicators'].iloc[i] for i in range(n_players)])
    prob += team_0_players >= 5
    prob += team_1_players >= 5
    
    # Credit constraint
    prob += lpSum([x[i] * df['Credits_x'].iloc[i] for i in range(n_players)]) <= 100
    
    # Playing 11 constraint
    # prob += lpSum([x[i] * df['IsPlaying_x'].iloc[i] for i in range(n_players)]) >= K/100 * 11
    
    # Solv
    prob.solve(solver)
    
    # Get selected players
    selected = pd.Series([0] * n_players)
    for i in range(n_players):
        if value(x[i]) == 1:
            selected.iloc[i] = 1
            
    # Return the filtered DataFrame instead of just the selection series
    return df[selected == 1].copy()
if __name__=='__main__':
    file='data/SquadPlayerNames - ipl.csv'
    
    # Dictionary mapping datetime objects to URLs
    match_schedule = {
        datetime.datetime(2025, 4, 17, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115165/dc-vs-rr-32nd-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 18, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115174/rcb-vs-pbks-34th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 19, 15, 30): "https://www.cricbuzz.com/cricket-match-facts/115176/gt-vs-dc-35th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 19, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115183/rr-vs-lsg-36th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 20, 15, 30): "https://www.cricbuzz.com/cricket-match-facts/115192/pbks-vs-rcb-37th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 20, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115201/mi-vs-csk-38th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 21, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115210/kkr-vs-gt-39th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 22, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115212/lsg-vs-dc-40th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 23, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115221/srh-vs-mi-41st-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 24, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115230/rcb-vs-rr-42nd-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 25, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115239/csk-vs-srh-43rd-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 26, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115248/kkr-vs-pbks-44th-match-indian-premier-league-2025", 
        datetime.datetime(2025, 4, 27, 15, 30): "https://www.cricbuzz.com/cricket-match-facts/115255/mi-vs-lsg-45th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 27, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115257/dc-vs-rcb-46th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 28, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115266/rr-vs-gt-47th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 29, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115275/dc-vs-kkr-48th-match-indian-premier-league-2025",
        datetime.datetime(2025, 4, 30, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115282/csk-vs-pbks-49th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 1, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115291/rr-vs-mi-50th-match-indian-premier-league-2025",  
        datetime.datetime(2025, 5, 2, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115300/gt-vs-srh-51st-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 3, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115302/rcb-vs-csk-52nd-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 4, 15, 30): "https://www.cricbuzz.com/cricket-match-facts/115309/kkr-vs-rr-53rd-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 4, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115318/pbks-vs-lsg-54th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 5, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115327/srh-vs-dc-55th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 6, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115336/mi-vs-gt-56th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 7, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115345/kkr-vs-csk-57th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 8, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115347/pbks-vs-dc-58th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 9, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115354/lsg-vs-rcb-59th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 10, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115356/srh-vs-kkr-60th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 11, 15, 30): "hhttps://www.cricbuzz.com/cricket-match-facts/115365/pbks-vs-mi-61st-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 11, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115372/dc-vs-gt-62nd-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 12, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115381/csk-vs-rr-63rd-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 13, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115390/rcb-vs-srh-64th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 14, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115392/gt-vs-lsg-65th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 15, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115401/mi-vs-dc-66th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 16, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115410/rr-vs-pbks-67th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 17, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115417/rcb-vs-kkr-68th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 18, 15, 30): "https://www.cricbuzz.com/cricket-match-facts/115426/gt-vs-csk-69th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 18, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115435/lsg-vs-srh-70th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 20, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115437/tbc-vs-tbc-qualifier-1-indian-premier-league-2025",
        datetime.datetime(2025, 5, 21, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115462/dc-vs-pbks-73rd-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 22, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115471/csk-vs-gt-74th-match-indian-premier-league-2025",
        datetime.datetime(2025, 5, 23, 19, 30): "https://www.cricbuzz.com/cricket-match-facts/115480/mi-vs-rr-75th-match-indian-premier-league-2025",
        
        
        
        
        

    }
    
    # Get current system datetime
    current_time = datetime.datetime.now()
    
    # Find closest datetime in dictionary
    closest_datetime = min(match_schedule.keys(), key=lambda x: abs((x - current_time).total_seconds()))
    
    # Select corresponding URL
    url = match_schedule[closest_datetime]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get team names from the match title
    match_title = soup.find('h1', class_='cb-nav-hdr cb-font-18 line-ht24')
    match_title_text = match_title.text.strip() if match_title else ''
    print(f"Match: {match_title_text}")

    # Extract team names from title
    teams = re.findall(r'(.*?) vs (.*?),', match_title_text)
    team1, team2 = teams[0] if teams else ('Team 1', 'Team 2')

    # Find toss information
    toss_label = soup.find(string="Toss:")
    toss_info = ""
    if toss_label:
        toss_element = toss_label.parent.find_next_sibling()
        if toss_element:
            toss_info = toss_element.text.strip()

    print(f"Toss info: {toss_info}")

    # Determine which team bats first
    batting_first = ""
    if toss_info:
        # Check if the toss winner chose to bat or bowl
        if "opt to bowl" in toss_info or "chose to bowl" in toss_info or "elected to bowl" in toss_info:
            # If toss winner chose to bowl, the other team bats first
            toss_winner = re.match(r'(.*?) won the toss', toss_info)
            if toss_winner:
                toss_winner_team = toss_winner.group(1)
                batting_first = team2 if toss_winner_team == team1 else team1
        elif "opt to bat" in toss_info or "chose to bat" in toss_info or "elected to bat" in toss_info:
            # If toss winner chose to bat, they bat first
            toss_winner = re.match(r'(.*?) won the toss', toss_info)
            if toss_winner:
                batting_first = toss_winner.group(1)

    # Extract venue and time
    venue_label = soup.find(string="Venue:")
    venue = venue_label.parent.find_next_sibling().text.strip() if venue_label else "Not found"

    time_label = soup.find(string="Time:")
    match_time = time_label.parent.find_next_sibling().text.strip() if time_label else "Not found"

    team_name_map = {'Mumbai Indians': 'MI', 'Royal Challengers Bengaluru': 'RCB', 'Chennai Super Kings': 'CHE', 'Delhi Capitals': 'DC', 'Gujarat Titans': 'GT', 'Kolkata Knight Riders': 'KKR', 'Lucknow Super Giants': 'LSG', 'Punjab Kings': 'PBKS', 'Rajasthan Royals': 'RR', 'Sunrisers Hyderabad': 'SRH'}
    team1 = team_name_map.get(team1, team1)
    team2 = team_name_map.get(team2, team2)
    batting_first = team_name_map.get(batting_first, batting_first)
    venue_map = {'Wankhede Stadium, Mumbai': 'Wankhede Stadium', 'M.Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium, Bengaluru', 'MA Chidambaram Stadium, Chennai': 'MA Chidambaram Stadium, Chepauk, Chennai', 'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium, Delhi', 'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium, Ahmedabad', 'Eden Gardens, Kolkata': 'Eden Gardens, Kolkata', 'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium, Jaipur', 'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium, Hyderabad', 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur, Chandigarh': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur', 'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium, Dharamsala', 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow'}
    venue = venue_map.get(venue, venue)

    # Print the results
    print(f"\nResults:")
    print(f"Team 1: {team1}")
    print(f"Team 2: {team2}")
    print(f"Venue: {venue}")
    print(f"Time: {match_time}")
    print(f"Batting first: {batting_first}")
    pop_inf(file,venue,team_dict[batting_first],datetime.datetime.now()) 
    player_df=pd.read_csv('data/inference/players.csv')
    matchups(player_df)
    matchups_final=generate_batsman_vs_opposition_metrics('data/inference/enhanced_matchups.csv','data/enhanced_matchups.csv', 'data/inference/match_batsman_details.csv','data/player_info.csv','24-04-2025')
    match_batsman_details=pd.read_csv('data/match_batsman_details.csv')
    match_bowler_details=pd.read_csv('data/match_bowler_details.csv')
    venue_player_df=pd.read_csv('data/venue_player_df.csv')
    match_bowler_details_venue=pd.read_csv('data/match_bowler_details_venue.csv')
    player_fielding=pd.read_csv('data/player_fielding.csv')
    final_matchups=pd.read_csv('data/final_matchups.csv')
    play_lst=[i for i in list(venue_player_df) if i not in list(match_batsman_details)]
    batter=pd.concat([match_batsman_details,venue_player_df[play_lst],final_matchups],axis=1)
    bowler=pd.concat([match_bowler_details,match_bowler_details_venue,final_matchups],axis=1)
    batter.to_csv('batter.csv')
    bowler.to_csv('bowler.csv')
    lst_batter=[i for i in list(batter) if 'T20_match' not in i and pd.Series(batter[i]).dtype!='O' and 'date' not in i and 'match_id' not in i and 'Unnamed' not in i]
    lst_bowler=[i for i in list(bowler) if 'T20_match' not in i and pd.Series(bowler[i]).dtype!='O' and 'date' not in i and 'match_id' not in i and 'Unnamed' not in i]
    lst_fielder=[i for i in list(player_fielding) if 'T20_match' not in i and pd.Series(player_fielding[i]).dtype!='O' and 'date' not in i and 'match_id' not in i and 'Unnamed' not in i]

    # Define feature columns for each target
    feature_columns_dict = {
        'T20_match_runs':lst_batter,
        'T20_match_4s':lst_batter,
        'T20_match_6s':lst_batter,
        'T20_match_strike_rate': lst_batter,
        'T20_match_wickets': lst_bowler,
        'T20_match_economy': lst_bowler,
        'T20_match_bowledlbw': lst_bowler,
        'T20_match_maidens': lst_bowler,
        'T20_match_balls_involved':lst_bowler,
        'T20_match_runouts': lst_fielder,
        'T20_match_catches': lst_fielder,
        'T20_match_stumpings': lst_fielder}

    # Define target columns for each player type
    target_columns_dict = {
    # T20 statistics
    'T20_batting': ['T20_match_runs', 'T20_match_4s', 'T20_match_6s', 'T20_match_strike_rate'],
    'T20_bowling': ['T20_match_wickets', 'T20_match_bowledlbw', 'T20_match_economy', 'T20_match_maidens','T20_match_balls_involved'],
    'T20_fielding': ['T20_match_runouts', 'T20_match_catches', 'T20_match_stumpings'],
    }
    matchups_final['player_name']=matchups_final['batsman']
    merged_df = pd.merge(pd.DataFrame(player_df['player_name']), matchups_final, on='player_name', how='inner')
    match_batsman_details=pd.read_csv('data/inference/match_batsman_details.csv')
    match_bowler_details=pd.read_csv('data/inference/match_bowler_details.csv')
    venue_player_df=pd.read_csv('data/inference/venue_player_df.csv')
    match_bowler_details_venue=pd.read_csv('data/inference/match_bowler_details_venue.csv')
    player_fielding_2=pd.read_csv('data/inference/fielder.csv')

    batter_2=pd.concat([match_batsman_details,venue_player_df[play_lst], merged_df],axis=1)
    bowler_2=pd.concat([match_bowler_details,match_bowler_details_venue,merged_df],axis=1)
    files_dict={
    # T20 statistics
    'T20_batting': [batter,batter_2],
    'T20_bowling': [bowler,bowler_2],
    'T20_fielding':[player_fielding,player_fielding_2]
    }
    print(bowler_2)
    for i in ['T20_batting','T20_fielding']:
     for j in target_columns_dict[i]:
      model = pickle.load(open('data/pickles/'+j+'.pkl', "rb"))  
      print(j)
      player_df[j]=model.predict(np.array(files_dict[i][1][feature_columns_dict[j]].fillna(0).replace([np.inf,-np.inf],0)))
    lst_bowler_venue=[i for i in list(match_bowler_details_venue) if 'T20_match' not in i and pd.Series(bowler[i]).dtype!='O' and 'date' not in i and 'match_id' not in i and 'Unnamed' not in i]
    bowler_2['inning_no']=1-batter_2['inning_no']
    print(bowler_2.shape)
    print(match_bowler_details_venue.shape)
    print(match_bowler_details_venue)
    for j in target_columns_dict['T20_bowling']:
        model=load_model('data/pickles/'+j+'.keras',custom_objects={"BottleneckResidualBlock": BottleneckResidualBlock})

        player_df[j]=model.predict([np.array(bowler_2[lst_bowler]),np.array(match_bowler_details_venue[lst_bowler_venue])])
    final_df=process_predictions(player_df,player_df)
    final_team=optimize_team(final_df)
    team1=[]
    team2=[]
    print(final_team)
    final_team=final_team.reset_index()
    teams=final_team['Team_x'].unique()
  

    for i in range(len(final_team)):
        print(team1)
        print(team2)
        if final_team['Team_x'][i]==teams[0]:
            
             if team1==[] and (final_team['Player Type_x'][i]=='BAT' or final_team['Player Type_x'][i]=='ALL'):
                team1.append(final_team['player_name'][i])
             elif final_team['scores'][i]>=final_team[final_team['player_name']==team1[0]]['scores'].values[0] and final_team['Player Type_x'][i]=='BAT' or final_team['Player Type_x'][i]=='ALL':
                team1=[final_team['player_name'][i]]
            
        else:
            if team2==[] and final_team['Player Type_x'][i]=='BAT' or final_team['Player Type_x'][i]=='ALL':
                team2.append(final_team['player_name'][i])
            elif final_team['scores'][i]>=final_team[final_team['player_name']==team2[0]]['scores'].values[0] and final_team['Player Type_x'][i]=='BAT' or final_team['Player Type_x'][i]=='ALL':
                team2=[final_team['player_name'][i]]
    c=1
    if final_team[final_team['player_name']==team1[0]]['scores'].values[0]>=final_team[final_team['player_name']==team2[0]]['scores'].values[0] :
        c=0
    captain=[]
    for i in range(len(final_team)):
        if c==0 and final_team['player_name'][i]==team1[0]:
            captain.append('C')
        elif c==0 and final_team['player_name'][i]==team2[0]:
             captain.append('VC')
        elif c==1 and final_team['player_name'][i]==team1[0]:
            captain.append('VC')
        elif c==1 and final_team['player_name'][i]==team2[0]:
             captain.append('C')
        else:
            captain.append('NC')
    print(pd.DataFrame({'player_name':final_team['Player Name'],'captain':captain,'type':final_team['Player Type_x']}))





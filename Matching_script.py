import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA


def Match_finder(Mentor_json,Mentees_json,model_file_location):
    model=pickle.load(open(model_file_location,'rb'))
    
    def data_preprocessing(data):
        data=data.drop(labels=['id','name','rollNo','peerID','post','hosteller'],axis=1)
        
        def cleaning_language_data(values):
            v=str(values)
            v=v.strip()
            s=re.sub(pattern="[^\w\s\,\/\+]",repl="",string=v)
            s=s.lower()
            res = re.split(', |/', s)
            count=0

        
            x=' '.join(res)
            x=x.replace('c++','chigh')
            x=x.replace('c','clow')
            x=x.replace('c#','cmedium')
            x=x.replace('clowhigh','chigh')
            x=x.replace('clow#','cmedium')
            x=x.replace('no preferenclowe','no prefence')
            return x

        def cleaning_domain_data(values):
            v=str(values)
            v=v.strip()
            s=re.sub(pattern="[^\w\s\,\/\+]",repl="",string=v)
            s=s.lower()
            res = re.split(', |/', s)
            count=0
            x=' '.join(res)
    
            return x
    
    
        for name,values in data.items():
            if name=='domains':
                if pd.api.types.is_string_dtype(values):
                    count=0
                    for i in values:
                        data.iloc[count,0]=cleaning_domain_data(i)
                        count+=1
            elif name=='languages':
                if pd.api.types.is_string_dtype(values):
                    count=0
                    for i in values:
                        data.iloc[count,1]=cleaning_language_data(i)
                        count+=1
            elif name=='branch':
                data['branch']=data['branch'].str.lower()
                branch_mapping={'cse-1':'cse','cse-2':'cse','it-1':'it','it-2':'it','ece':'ece','mae':'mae','cse':'cse','it':'it','eee':'eee','mechanical':'mechanical','civil':'civil'}
                data['branch']=data['branch'].map(branch_mapping)
        
    
        return data
    
    def perfect_match_process(full_mentor_df,processed_mentor_df,full_mentee_df,processed_mentee_df):
        domain_score=5
        language_score=1
        branch_score=0.5
        match_score=[]
        
        mentor_matched_id=[]
        mentee_matched_id=[]
        
        
        fk=processed_mentor_df.copy()
        nik=[]
        for i in np.random.randint(1,6,size=(fk.shape[0],1)):
            nik.append(i[0])
        fk['experience_level']=nik
        predict_branch_data=processed_mentee_df
        for i in range(len(predict_branch_data)):
            target_matched_df=fk[fk['target']==predict_branch_data.iloc[i,:]['target']]
#             print(fk.shape)
            temp_mentee=predict_branch_data.iloc[i,:-1]
            temp_mentee_domain=temp_mentee['domains'].lower().split()
            temp_mentee_language=temp_mentee['languages'].lower().split()
            temp_mentee_branch=temp_mentee['branch'].split()
 
            for j in range(len(target_matched_df)):
        
                each_score=0
                temp_df=np.array(target_matched_df.iloc[j,:-2])
      
                st=''
                for k in range(len(temp_df)):
                    st+=' '+temp_df[k]
        
        
        
                for l in temp_mentee_domain:
                    if l in st:
                        each_score+=domain_score
        
                for m in temp_mentee_language:
                    if m in st:
                        each_score+=language_score

        
                for n in temp_mentee_branch:
                    if n in st:
                        each_score+=branch_score
                match_score.append(each_score)
        

                del st
            try:
                target_matched_df['match_score']=match_score
                best_match=target_matched_df[target_matched_df['match_score']==max(match_score)]
       
                if len(best_match)>1:
                    th=fk.sort_values('experience_level',ascending=False).loc[best_match.index]
                
                    ind=th.index[0]
                
                    mentor_matched_id.append(full_mentor_df['id'].loc[ind])
                    fk=fk.drop(ind,axis=0)
        
        
                else:
                    perfect_matched_pair_index=best_match.index[0]
#                 print(perfect_matched_pair_index)
    
                    mentor_matched_id.append(full_mentor_df['id'].loc[perfect_matched_pair_index])
        
                    fk=fk.drop(target_matched_df[target_matched_df['match_score']==max(match_score)].index[0],axis=0)

                mentee_matched_id.append(full_mentee_df['id'].loc[i])
                match_score.clear()
            except:
                print('Not Found')
        final_perfect_matched=pd.DataFrame()
        final_perfect_matched['mentor_id']=mentor_matched_id
        final_perfect_matched['mentees_id']=mentee_matched_id
        
        return final_perfect_matched
        
                                                
    
    
    def Mentor_Matching(Mentor_json,Mentees_json,model):
        Mentor_df_json=pd.read_json(Mentor_json)
        Mentor_df=pd.DataFrame(Mentor_df_json.users.values.tolist())
        
        data_with_branch=Mentor_df.copy()
        data=Mentor_df.copy()
        
        
#         data=data.drop(labels=['id','name','rollNo','peerID','post','hosteller'],axis=1)
        df=pd.DataFrame({'domains':['[Web Development, App Development, Machine Learning, IOT, BlockChain, AR/VR, Game Development, Cloud Engineering, Competitive Programming, Cyber Security, Open Source]'],'languages':['[Java, Python, C/C++, No Preference]'],'branch':['cse']})
        data=data.append(df,ignore_index=True)
        mentor_df_pre=data_preprocessing(data)
#         nik=[]
#         for i in np.random.randint(1,6,size=(data.shape[0],1)):
#             nik.append(i[0])
#         mentor_df_pre['experience_level']=nik
        np.random.seed(10)
        domains_vector=CountVectorizer()
        domains_vec_val=domains_vector.fit_transform(mentor_df_pre['domains'])
        df_domain_wrds=pd.DataFrame(domains_vec_val.toarray(),columns=domains_vector.get_feature_names())
        np.random.seed(11)
        languages_vector=CountVectorizer()
        languages_vec_val=languages_vector.fit_transform(mentor_df_pre['languages'])
        df_lang_wrds=pd.DataFrame(languages_vec_val.toarray(),columns=languages_vector.get_feature_names())
        final_df=pd.concat([df_domain_wrds,df_lang_wrds],axis=1)
#         final_df=final_df.drop(['domains','languages'],axis=1)
        final_df=final_df.drop(index=final_df.shape[0]-1,axis=0)
        pca=PCA(n_components=2,random_state=30)
        X=pca.fit_transform(final_df)
        values=model.predict(X)
        Mentor_df_pre=mentor_df_pre.drop(index=mentor_df_pre.shape[0]-1,axis=0)
        Mentor_df_pre['target']=values
        # For Mentor Matching
        Mentor_matching_df_json=pd.read_json(Mentees_json)
        predict_df=pd.DataFrame(Mentor_matching_df_json.users.values.tolist())
        predict_data=predict_df.copy()
        
#         print(predict_data.shape)
#         predict_data_index=predict_data.isna().index
#         predict_data=predict_data.drop(predict_data_index,axis=0)
#         print(predict_data.shape)
        
        predict_data=data_preprocessing(predict_data)
        predict_domains=domains_vector.transform(predict_data['domains'])
        predict_domain_wrds=pd.DataFrame(predict_domains.toarray(),columns=domains_vector.get_feature_names())
        predict_languages=languages_vector.transform(predict_data['languages'])
        predict_lang_wrds=pd.DataFrame(predict_languages.toarray(),columns=languages_vector.get_feature_names())
        final_df=pd.concat([predict_domain_wrds,predict_lang_wrds],axis=1)
#         final_df=final_df.drop(['domains','languages'],axis=1)
        Predicted_X=pca.transform(final_df)
        predicted_target=model.predict(Predicted_X)
        predict_data['target']=predicted_target
        matched_pairs_id=perfect_match_process(Mentor_df,Mentor_df_pre,predict_df,predict_data)
            
            
            
        return matched_pairs_id
    
    matching_df=Mentor_Matching(Mentor_json,Mentees_json,model)
    return matching_df
    
        
        

        
        
        
        
        
        
        
        
        
        
        
    
import pickle
import pandas as pd


def more_data_uploading(data_name, type_data_name):

    if data_name =='Flavus':

        columns_metadata = ['jour', 'nom_clone', 'plaque', 'souche']
        train_metadata_name = './Flavus_data_methodo/Flavus_metadata_train'
        train_metadata = pd.DataFrame(data=pickle.loads(open(train_metadata_name, 'rb').read()))
        train_metadata.columns = columns_metadata

        test_metadata_name = './Flavus_data_methodo/Flavus_metadata_test'
        test_metadata = pd.DataFrame(data=pickle.loads(open(test_metadata_name, 'rb').read()))
        test_metadata.columns = ['maldi'] + columns_metadata

        if type_data_name == 'Intensite_interp_4':
            columns_data_treated = ['Intensite smooth baseline als interp reduc 4']
            train_data_treated_name = './Flavus_data_methodo/smooth_basl_interp_4_train_flavus'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_data_treated_name = './Flavus_data_methodo/smooth_basl_interp_4_test_flavus'
            test_data_treated = pd.DataFrame(data=pickle.loads(open(test_data_treated_name, 'rb').read()))
            test_data_treated.columns = columns_data_treated

        #elif type_data_name == 'other_type_spectra':

        else:
            raise ValueError("type_data_name unknown.")
        
        train_data = pd.concat([train_metadata, train_data_treated], axis=1)
        
        test_data = pd.concat([test_metadata, test_data_treated], axis=1)

        return train_data, test_data
    
    elif data_name =='Candida_parapsilosis':

        columns_metadata = ['maldi', 'nom_clone', 'souche', 'milieu']
        train_metadata_name = './CP_data_methodo/CP_metadata_train'
        train_metadata = pd.DataFrame(data=pickle.loads(open(train_metadata_name, 'rb').read()))
        train_metadata.columns = columns_metadata

        test_metadata_name = './CP_data_methodo/CP_metadata_test'
        test_metadata = pd.DataFrame(data=pickle.loads(open(test_metadata_name, 'rb').read()))
        test_metadata.columns =  columns_metadata

        if type_data_name == 'Intensite_interp_4':
            columns_data_treated = ['Intensite smooth baseline als interp reduc 4']
            train_data_treated_name = './CP_data_methodo/smooth_basl_interp_4_train_CP'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_data_treated_name = './CP_data_methodo/smooth_basl_interp_4_test_CP'
            test_data_treated = pd.DataFrame(data=pickle.loads(open(test_data_treated_name, 'rb').read()))
            test_data_treated.columns = columns_data_treated

        #elif type_data_name == 'other_type_spectra':

        else:
            raise ValueError("type_data_name unknown.")
        
        train_data = pd.concat([train_metadata, train_data_treated], axis=1)
        
        test_data = pd.concat([test_metadata, test_data_treated], axis=1)

        print('WARNING < ! > : Target feature is already encode !')

        return train_data, test_data
        
        
    elif data_name =='Anophele_age':
        columns_metadata = ['jours','anatomie', 'type_donnees', 'sous_fichier','pos_plaque']
        train_metadata_name = './Anopheles_data_methodo/anophele_age_metadata_train'
        train_metadata = pd.DataFrame(data=pickle.loads(open(train_metadata_name, 'rb').read()))
        train_metadata.columns = columns_metadata

        test_pauline_metadata_name = './Anopheles_data_methodo/anophele_age_metadata_test_pauline'
        test_pauline_metadata = pd.DataFrame(data=pickle.loads(open(test_pauline_metadata_name, 'rb').read()))
        test_pauline_metadata.columns =  columns_metadata

        test_noemie_metadata_name = './Anopheles_data_methodo/anophele_age_metadata_test_noemie'
        test_noemie_metadata = pd.DataFrame(data=pickle.loads(open(test_noemie_metadata_name, 'rb').read()))
        test_noemie_metadata.columns =  columns_metadata

        if type_data_name == 'Intensite_interp_4':
            columns_data_treated = ['Intensite smooth baseline als interp reduc 4']
            train_data_treated_name = './Anopheles_data_methodo/smooth_basl_interp_4_train_ano_age'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_pauline_data_treated_name = './Anopheles_data_methodo/smooth_basl_interp_4_test_pauline_ano_age'
            test_pauline_data_treated = pd.DataFrame(data=pickle.loads(open(test_pauline_data_treated_name, 'rb').read()))
            test_pauline_data_treated.columns = columns_data_treated

            test_noemie_data_treated_name = './Anopheles_data_methodo/smooth_basl_interp_4_test_noemie_ano_age'
            test_noemie_data_treated = pd.DataFrame(data=pickle.loads(open(test_noemie_data_treated_name, 'rb').read()))
            test_noemie_data_treated.columns = columns_data_treated


        elif type_data_name == 'Intensite_basl_interp_5':
            columns_data_treated = ['Intensite baseline als interp reduc 5']
            train_data_treated_name = './Anopheles_data_methodo/basl_interp_5_train_ano_age'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_pauline_data_treated_name = './Anopheles_data_methodo/basl_interp_5_test_1_ano_age'
            test_pauline_data_treated = pd.DataFrame(data=pickle.loads(open(test_pauline_data_treated_name, 'rb').read()))
            test_pauline_data_treated.columns = columns_data_treated

            test_noemie_data_treated_name = './Anopheles_data_methodo/basl_interp_5_test_2_ano_age'
            test_noemie_data_treated = pd.DataFrame(data=pickle.loads(open(test_noemie_data_treated_name, 'rb').read()))
            test_noemie_data_treated.columns = columns_data_treated

        elif type_data_name == 'Traitement_all':
            columns_data_treated = ['Traitement all']
            train_data_treated_name = './Anopheles_data_methodo/Traitement_all_train_ano_age'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_pauline_data_treated_name = './Anopheles_data_methodo/Traitement_all_test_1_ano_age'
            test_pauline_data_treated = pd.DataFrame(data=pickle.loads(open(test_pauline_data_treated_name, 'rb').read()))
            test_pauline_data_treated.columns = columns_data_treated

            test_noemie_data_treated_name = './Anopheles_data_methodo/Traitement_all_test_2_ano_age'
            test_noemie_data_treated = pd.DataFrame(data=pickle.loads(open(test_noemie_data_treated_name, 'rb').read()))
            test_noemie_data_treated.columns = columns_data_treated

        #elif type_data_name == 'other_type_spectra':

        else:
            raise ValueError("type_data_name unknown.")
        
        train_data = pd.concat([train_metadata, train_data_treated], axis=1)
        
        test_pauline_data = pd.concat([test_pauline_metadata, test_pauline_data_treated], axis=1)

        test_noemie_data = pd.concat([test_noemie_metadata, test_noemie_data_treated], axis=1)

        return train_data, test_pauline_data, test_noemie_data
        

    elif data_name =='Anophele_identif':
        
        columns_metadata = ['espece','anatom', 'provenance', 'sous_fichier','sous_sous_fichier']
        train_metadata_name = './Anopheles_data_methodo/Ano_ident_metadata_train'
        train_metadata = pd.DataFrame(data=pickle.loads(open(train_metadata_name, 'rb').read()))
        train_metadata.columns = columns_metadata

        test_metadata_name = './Anopheles_data_methodo/Ano_ident_metadata_test'
        test_metadata = pd.DataFrame(data=pickle.loads(open(test_metadata_name, 'rb').read()))
        test_metadata.columns =  columns_metadata

        if type_data_name == 'Intensite_interp_4':
            columns_data_treated = ['Intensite smooth baseline als interp reduc 4']
            train_data_treated_name = './Anopheles_data_methodo/smooth_basl_interp_4_train_ano_ident'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_data_treated_name = './Anopheles_data_methodo/smooth_basl_interp_4_test_ano_ident'
            test_data_treated = pd.DataFrame(data=pickle.loads(open(test_data_treated_name, 'rb').read()))
            test_data_treated.columns = columns_data_treated

        elif type_data_name == 'Intensite_basl_interp_5':
            columns_data_treated = ['Intensite baseline als interp reduc 5']
            train_data_treated_name = './Anopheles_data_methodo/basl_interp_5_train_ano_ident'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_data_treated_name = './Anopheles_data_methodo/basl_interp_5_test_ano_ident'
            test_data_treated = pd.DataFrame(data=pickle.loads(open(test_data_treated_name, 'rb').read()))
            test_data_treated.columns = columns_data_treated

        #elif type_data_name == 'other_type_spectra':

        else:
            raise ValueError("type_data_name unknown.")
        
        train_data = pd.concat([train_metadata, train_data_treated], axis=1)
        
        test_data = pd.concat([test_metadata, test_data_treated], axis=1)

        
        return train_data, test_data


    elif data_name =='MABSC_coh':
        
        columns_metadata = ['espece', 'souche', 'data_base', 'resistance', 'pos_plaque', 'Tirage']
        train_metadata_name = './MABSC/mabsc_metadata_coh_train'
        train_metadata = pd.DataFrame(data=pickle.loads(open(train_metadata_name, 'rb').read()))
        train_metadata.columns = columns_metadata

        test_metadata_name = './MABSC/mabsc_metadata_coh_test'
        test_metadata = pd.DataFrame(data=pickle.loads(open(test_metadata_name, 'rb').read()))
        test_metadata.columns =  columns_metadata

        if type_data_name == 'Intensite_interp_4':
            columns_data_treated = ['Intensite smooth baseline als interp reduc 4']
            train_data_treated_name = './MABSC/smooth_basl_interp_4_train_mabsc'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_data_treated_name = './MABSC/smooth_basl_interp_4_test_mabsc'
            test_data_treated = pd.DataFrame(data=pickle.loads(open(test_data_treated_name, 'rb').read()))
            test_data_treated.columns = columns_data_treated

        elif type_data_name == 'Intensite_interp_5':
            columns_data_treated = ['Intensite smooth baseline als interp reduc 5']
            train_data_treated_name = './MABSC/smooth_basl_interp_5_train_mabsc'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_data_treated_name = './MABSC/smooth_basl_interp_5_test_mabsc'
            test_data_treated = pd.DataFrame(data=pickle.loads(open(test_data_treated_name, 'rb').read()))
            test_data_treated.columns = columns_data_treated

        elif type_data_name == 'Intensite_basl_interp_5':
            columns_data_treated = ['Intensite baseline als interp reduc 5']
            train_data_treated_name = './MABSC/basl_interp_5_train_mabsc'
            train_data_treated = pd.DataFrame(data=pickle.loads(open(train_data_treated_name , 'rb').read()))
            train_data_treated.columns = columns_data_treated

            test_data_treated_name = './MABSC/basl_interp_5_test_mabsc'
            test_data_treated = pd.DataFrame(data=pickle.loads(open(test_data_treated_name, 'rb').read()))
            test_data_treated.columns = columns_data_treated

        #elif type_data_name == 'other_type_spectra':

        else:
            raise ValueError("type_data_name unknown.")
        
        train_data = pd.concat([train_metadata, train_data_treated], axis=1)
        
        test_data = pd.concat([test_metadata, test_data_treated], axis=1)

        
        return train_data, test_data
        
    
    else:
        raise ValueError("Dataset name unknown.")




def data_uploading(name):
    if name=='Flavus':
        train_data_name = 'Data_Flavus_align'
        train_data = pd.DataFrame(data=pickle.loads(open(train_data_name, 'rb').read()))
        train_data.columns = ['jour', 'nom_clone', 'plaque', 'souche','Traitement all', 'Intensite align msiwarp']

        test_data_name = 'Data_test_Flavus_align'
        test_data = pd.DataFrame(data=pickle.loads(open(test_data_name, 'rb').read()))
        test_data.columns = ['maldi', 'jour', 'nom_clone','plaque', 'souche', 'Traitement all','Intensite align msiwarp']
        
        return train_data, test_data
    
    elif name=='Candida_parapsilosis':
        columns = ['maldi', 'nom_clone', 'souche', 'milieu', 'Traitement all','Intensite align msiwarp']
        train_data_name = 'Data_train_CP_align'
        train_data = pd.DataFrame(data=pickle.loads(open(train_data_name, 'rb').read()))
        train_data.columns = columns

        test_data_name = 'Data_test_CP_align'
        test_data = pd.DataFrame(data=pickle.loads(open(test_data_name, 'rb').read()))
        test_data.columns = columns

        print('WARNING < ! > : Target feature is already encode !')
        
        return train_data, test_data
        
    elif name=='Anophele_age':
        columns = ['jours', 'anatomie', 'type_donnees', 'sous_fichier', 'pos_plaque', 'Intensite align msiwarp',
       'Partie anatomique']
        DATA_SAMPLE_1 = 'Data_anophele_align_pauline_train'
        data_extract_1 = pickle.loads(open(DATA_SAMPLE_1, 'rb').read())
        Data_1 = pd.DataFrame(data=data_extract_1)
        Data_1.columns = columns
        
        DATA_SAMPLE_2 = 'Data_anophele_align_noemie_train'
        data_extract_2 = pickle.loads(open(DATA_SAMPLE_2, 'rb').read())
        Data_2 = pd.DataFrame(data=data_extract_2)
        Data_2.columns = columns
        
        Data = pd.concat([Data_1, Data_2])
        Data = Data.sample(frac=1)
        
        DATA_SAMPLE_3 = 'Data_anophele_align_pauline_test'
        data_extract_3 = pickle.loads(open(DATA_SAMPLE_3, 'rb').read())
        Data_test_3 = pd.DataFrame(data=data_extract_3)
        Data_test_3.columns = columns


        DATA_SAMPLE = 'Data_anophele_align_noemie_test'
        data_extract = pickle.loads(open(DATA_SAMPLE, 'rb').read())
        Data_test_4 = pd.DataFrame(data=data_extract)
        Data_test_4.columns = columns
        
        return Data, Data_test_3, Data_test_4


    elif name=='Anophele_age_80_20':
        columns = ['jours', 'anatomie', 'type_donnees', 'sous_fichier', 'pos_plaque', 'Masse', 'Intensite', 'Partie anatomique', 'Traitement all', 'Masse mz msiwarp', 'Intensite align msiwarp' ]
        
        DATA_SAMPLE_train_pauline = './anopheles_age_new_split_80_20/anophele_age_80_20_train_pauline_align_new'
        data_extract_train_pauline = pickle.loads(open(DATA_SAMPLE_train_pauline, 'rb').read())
        Data_train_pauline = pd.DataFrame(data=data_extract_train_pauline)
        Data_train_pauline.columns = ['jours', 'anatomie', 'type_donnees', 'sous_fichier', 'pos_plaque','Partie anatomique', 'Traitement all','Intensite align msiwarp' ]
        Data_train_pauline = Data_train_pauline[['jours', 'anatomie', 'type_donnees', 'sous_fichier', 'pos_plaque', 'Intensite align msiwarp',
       'Partie anatomique']]
        
        DATA_SAMPLE_train_noemie = './anopheles_age_new_split_80_20/anophele_age_80_20_train_noemie_align_new'
        data_extract_train_noemie = pickle.loads(open(DATA_SAMPLE_train_noemie, 'rb').read())
        Data_train_noemie = pd.DataFrame(data=data_extract_train_noemie)
        Data_train_noemie.columns = ['jours', 'anatomie', 'type_donnees', 'sous_fichier', 'pos_plaque','Partie anatomique', 'Traitement all','Intensite align msiwarp' ]
        Data_train_noemie = Data_train_noemie[['jours', 'anatomie', 'type_donnees', 'sous_fichier', 'pos_plaque', 'Intensite align msiwarp',
       'Partie anatomique']]
        
        Data = pd.concat([Data_train_pauline, Data_train_noemie])
        Data = Data.sample(frac=1)

        del data_extract_train_pauline, data_extract_train_noemie, Data_train_pauline, Data_train_noemie
        

        DATA_SAMPLE_test_1 = './anopheles_age_new_split_80_20/anophele_age_80_20_pauline_align'
        data_extract_test_1 = pickle.loads(open(DATA_SAMPLE_test_1, 'rb').read())
        Data_test_1 = pd.DataFrame(data=data_extract_test_1)
        Data_test_1.columns = columns
        Data_test_1 = Data_test_1[['jours', 'anatomie', 'type_donnees', 'sous_fichier', 'pos_plaque', 'Intensite align msiwarp',
       'Partie anatomique']]
        

        DATA_SAMPLE_test_2 = './anopheles_age_new_split_80_20/anophele_age_80_20_noemie_align'
        data_extract_test_2 = pickle.loads(open(DATA_SAMPLE_test_2, 'rb').read())
        Data_test_2 = pd.DataFrame(data=data_extract_test_2)
        Data_test_2.columns = columns
        Data_test_2 = Data_test_2[['jours', 'anatomie', 'type_donnees', 'sous_fichier', 'pos_plaque', 'Intensite align msiwarp',
       'Partie anatomique']]
        
        return Data, Data_test_1, Data_test_2
    

    elif name=='Anophele_identif':
        columns = ['sous_fichier', 'espece', 'anatom', 'provenance', 'sous_sous_fichier', 'Traitement all', 'Intensite align msiwarp']
        train_data_name = 'Ident_anophele_align_train'
        train_data = pd.DataFrame(data=pickle.loads(open(train_data_name, 'rb').read()))
        train_data.columns = columns

        test_data_name = 'Ident_anophele_align_test'
        test_data = pd.DataFrame(data=pickle.loads(open(test_data_name, 'rb').read()))
        test_data.columns = columns
        
        return train_data, test_data
    
    elif name=='MABSC_coh':
        columns = ['espece', 'souche', 'data_base', 'resistance', 'pos_plaque', 'Tirage','Traitement all', 'Intensite align msiwarp']
        train_data_name = 'mabsc_train_coh_07_2023_align'
        train_data = pd.DataFrame(data=pickle.loads(open(train_data_name, 'rb').read()))
        train_data.columns = columns

        test_data_name = 'mabsc_test_coh_07_2023_align'
        test_data = pd.DataFrame(data=pickle.loads(open(test_data_name, 'rb').read()))
        test_data.columns = columns
        
        return train_data, test_data
    else:
        raise ValueError("Dataset name unknown.")
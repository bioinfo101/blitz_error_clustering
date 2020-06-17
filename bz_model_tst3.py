#!/usr/bin/python3

import blitzlib as blz
from mstrio import microstrategy

import tkinter as tk
import PySimpleGUI as sg
import re
import nltk
import numpy as np
import pandas as pd
import os
import pickle
import warnings
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from datetime import datetime, timedelta
from operator import itemgetter
from scipy.spatial import distance

warnings.filterwarnings("ignore")

# MSTR REST API URL
# https://aqueduct-tech.customer.cloud.microstrategy.com/MicroStrategyLibrary/api
# https://aqueduct.microstrategy.com/MicroStrategyLibrary/api

baseURLs = ["env-141185.customer.cloud.microstrategy.com",
"aqueduct.microstrategy.com",
"aqueduct-tech.customer.cloud.microstrategy.com"]


# baseURLs = ["https://env-141185.customer.cloud.microstrategy.com/MicroStrategyLibrary/api",
# "https://aqueduct.microstrategy.com/MicroStrategyLibrary/api",
# "https://aqueduct-tech.customer.cloud.microstrategy.com/MicroStrategyLibrary/api"]


# MSTR project name
projName = "Rally Analytics"

fields = ('MSTR Login Name:', 'Password:', 'Blitz Label:', 'Test Date [format: YYYYMMDD]:', 'Test Site [hqt, ctc, uat, aba, or any 3 letters]:')
default_txt = ('mstr','tRLDF2NvOWrV', 'blitz_11.1.0100.0018', '20190205', 'hqt', '??????', '?')

# blitz_11.1.0200.0070

columns = ['BLITZ', 'PID', 'TIME STAMP', 'PRODUCT', 'ERROR', 'IS FATAL', 'EXTRA', 'UID', 'SID', 'OID', 'THR', 'KEYWORD','WEIGHT', 'IDENTICALS']
truncat = ['xml', 'XML', 'select', 'SELECT', 'MB', '[SimpleCSVParser]', 'SQL Statement:']
pa_columns = ['PID', 'TIME STAMP', 'PRODUCT', 'ERROR', 'IS FATAL', 'EXTRA', 'UID', 'SID', 'OID', 'THR', 'KEYWORD', 'WEIGHT', 'IDENTICALS', 'ROUNDED TIME', 'CPU', 'MEMORY']

value_list = [None] * 7

# value_list[]
#   0: username
#   1: password
#   2: blitz label
#   3: test date
#   4: test site
#   5: env id
#   6: node no

df_err = pd.DataFrame(columns=columns)
df_uniq_err = []
df_err_pa = pd.DataFrame(columns=pa_columns)

##############################################################
def close_window ():
    root.destroy()

##############################################################
def logSelect():
    global infilename

    print('\nStarting ' + '\x1b[6;30;42m' + ' STEP 2 ' + '\x1b[0m')
    infilename = askopenfilename(initialdir="/", title="Select Target DSSError log", filetypes=((".log", "*.log"), ("all files", "*.*")))
    print('Selected DSSError log name as input: ', infilename)
    print('\x1b[1;33m' + 'Done with [DSSError Log File Selection].' + '\x1b[0m')

##############################################################
def vecSelect():
    global vec_filename

    vec_filename = askopenfilename(initialdir="/", title="Select Term Vectorizer",
                                 filetypes=((".sav", "*.sav"), ("all files", "*.*")))
    print('Selected Term Vectorizer file as input: ', vec_filename)

##############################################################
def gallerySelect():
    global pty_filename

    pty_filename = askopenfilename(initialdir="/", title="Select Cluster Prototypes",
                                 filetypes=((".csv", "*.csv"), ("all files", "*.*")))
    print('Selected Gallery Prototype for matching: ', pty_filename)

##############################################################
def paSelect():
    global pafilename
    global envfilename

    print('\nStarting ' + '\x1b[6;30;42m' + ' STEP 5 ' + '\x1b[0m')
    pafilename = askopenfilename(initialdir="/", title="Select PA Data",
                                 filetypes=((".csv", "*.csv"), ("all files", "*.*")))
    envfilename = pafilename[:-4] + '_instance.csv'
    print('PA Metrics: ', pafilename)
    print('PA Environment: ', envfilename)
    print('\x1b[1;33m' + 'Done with [PA Data File Selection].' + '\x1b[0m')

##############################################################
def passwd_show():
    p = password.get() #get password from entry
    print(p)

##############################################################
def fetch(entries):
    global outfilename
    global baseURL
    global isLDAP
    global this_prefix

    print('\nStarting ' + '\x1b[6;30;42m' + ' STEP 1 ' + '\x1b[0m')
    url_sel = var0.get()
    print("Your selected MSTR URL:  " + str(url_sel))
    baseURL = 'https://'+ url_sel +'/MicroStrategyLibrary/api'
    print(baseURL)
    isLDAP = var2.get()
    filename_prefix = 'DSSErr'
    for entry in entries:
        field = entry[0]
        if entry ==1:
            text =  Entry(app, textvariable=field, show='*')
        else:
            text  = entry[1].get()
        # print('%s: "%s"' % (field, text))
        value_list[fields.index(field)] = text

    value_list[5] = default_txt[5]
    value_list[6] = default_txt[6]
    this_prefix = entry3.get()



    outfilename = filename_prefix + value_list[3]+ '_n' + value_list[6]+ '_' + value_list[5] + '_'+ value_list[4]

    outfilename = outfilename + '.csv'
    print('Output filename will look like: ' + outfilename)
    print('\x1b[1;34m' + 'Note that: the Node number and ENV_id will be automatically detected once DSSError log file is loaded in STEP 2' + '\x1b[0m')
    print('\x1b[1;34m' + 'the node number will match to an instance ID that will be used to extract the corresponding PA utilization metric in STEP 6' + '\x1b[0m')
    print('\x1b[1;33m' + 'Done with [Parameter Setting].' + '\x1b[0m')

##############################################################
def makeform(root, fields):

    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=40, text = field, anchor='w')
        ent = tk.Entry(row)

        temp = default_txt[fields.index(field)]

        ent.insert(0, temp)
        #
        lab.grid(row=2 + fields.index(field), column=0, columnspan=2, padx=6)
        row.grid(row=2 + fields.index(field), column=0, columnspan=2, padx=6, pady=12)
        ent.grid(row=2 + fields.index(field), column=2, columnspan=2)
        #
        entries.append((field, ent))
    return entries

##############################################################
def getAtribute(key, tline, startPOS, endPOS):

    whereATTR = tline[startPOS:endPOS].find(key)
    whereATTR = whereATTR + startPOS
    if (whereATTR > -1):
        rightqATTR = tline[whereATTR : whereATTR + endPOS].find(']')
        ATTRval = tline[whereATTR + len(key): whereATTR + rightqATTR]
    else:
        rightqATTR = 0
        ATTRval = ''

    return whereATTR, rightqATTR + whereATTR, ATTRval

##############################################################
def parse():
    global infilename
    global outfilename
    global df_err
    global df_uniq_err
    global envStr

    errorCount = -1
    currentLine = 1
    filename_prefix = 'DSSErr'
    isFound = False

    FMT1 = '%Y-%m-%d %H:%M:%S'
    # FMT1 = '%m/%d/%Y %H:%M'
    date_str = value_list[3][0:4] + '-'

    print('\nStarting ' + '\x1b[6;30;42m' + ' STEP 3 '+ '\x1b[0m')

    # find total line number
    nb_lines = sum(1 for line in open(infilename,encoding="utf8"))

    # Read keyword watchlist
    df_keyword = pd.read_csv("error_keywords.csv")
    nb_keyword = len(df_keyword)
    nexttoMainline = False

    # Identify env_id and node number = (1,2)
    fin = open(infilename, "r", encoding="utf8")
    tline = fin.readline()
    while not isFound:
        whereTIMESTAMP = tline[0:30].find(date_str)
        if whereTIMESTAMP >= 0:
            whereHOST, rightqHOST, HOSTval = getAtribute('[HOST:env-', tline, whereTIMESTAMP+24, whereTIMESTAMP+24+30)
            if not (HOSTval == []):
                whereLaio = HOSTval.find('laio')
                if  whereLaio >= 0:
                    # ENV_ID
                    value_list[5] = HOSTval[0:whereLaio]
                    # Node No
                    value_list[6] = HOSTval[whereLaio+4]
                    isFound = True
        tline = fin.readline()
    fin.close()
    envStr = 'env-' + HOSTval
    print('Enviroment ID: ' + envStr)

    outfilename = filename_prefix +  value_list[3] + 'n' + value_list[6] + '_' + value_list[5] + '_' + value_list[4]+ '.csv'
    print('Output filename: ', outfilename)

    # Open file
    fin = open(infilename, "r", encoding="utf8")
    tline = fin.readline()

    ary_err = []
    print("****** Parsing DSSError log text ... ")
    blz.tic()
    while tline:
        # Pop up a progress bar
        if ( currentLine % 200 ==1 ) or (currentLine+200 > nb_lines):
            sg.OneLineProgressMeter('Line Parsing', currentLine, nb_lines, 'key')

        isMainLine = False
        whereTIMESTAMP = tline[0:30].find(date_str)
        if whereTIMESTAMP >= 0:             # if datetime string can be found
            EXTRAval = ''
            rightTIMESTAMP = whereTIMESTAMP + 11
            whereERROR = tline[rightTIMESTAMP :].find('[Error]')    # Check if [Error} or [Fetal] exists
            whereFATAL = tline[rightTIMESTAMP :].find('[Fatal]')

            if ((whereERROR > -1) or (whereFATAL > -1)):
                isMainLine = True

                errorCount = errorCount + 1
                if (whereFATAL > -1):
                    whereERROR = whereFATAL

                # to find the absolute position of [ERROR] by dding the TIMESTAMP offset
                whereERROR = whereERROR + rightTIMESTAMP
                # record PID:
                wherePID, rightqPID, PIDval = getAtribute('[PID:', tline, rightTIMESTAMP, whereERROR)
                # record THR:
                whereTHR, rightqTHR, THRval = getAtribute('[THR:', tline,  rightqPID, whereERROR)
                # record PRODUCT
                wherePPRODUCT, rightqPRODUCT, PRODUCTval = getAtribute('[', tline, rightqTHR, whereERROR)
                # record UID
                whereUID, rightqUID, UIDval = getAtribute('[UID:', tline, whereERROR, whereERROR+50)
                # record SID
                whereSID, rightqSID, SIDval  = getAtribute('[SID:', tline, rightqUID, rightqUID + 50)
                # record OID
                if (whereSID > -1):
                    whereOID, rightqOID, OIDval = getAtribute('[OID:', tline, rightqSID, rightqSID + 50)
                else:
                    whereOID, rightqOID, OIDval = getAtribute('[OID:', tline, rightqUID, rightqUID + 100)
                # record TIME STAMP
                TIMESTAMPval = pd.to_datetime(tline[whereTIMESTAMP: whereTIMESTAMP + 19])
                # record ERROR
                if (whereOID == -1):
                    rightqERROR = whereERROR + 6
                    if (whereSID > -1):
                        # [To handle the case when SID exists]:
                        # 2018-05-14 20:35:34.745 [HOST:env-93835laio1use1][SERVER:CastorServer][PID:5949][THR:139643536082688]
                        # [Distribution Service][Error][UID:2ED12F4211E7409200000080EF755231]
                        # [SID:3723DC217D2D1C26E2AD86E25D5D1552] MSIDeliveryEngine::hDelivery(): Unknown Delivery Failed. Error string
                        # from ExecuteMultiProcess SSL Error: A failure in the SSL library occurred, usually a protocol error [Provider
                        # certificate may expire]. . <Subscription '' (ID = 00000000000000000000000000000000), Contact 'Monitoring, HeartBeat' (ID = A86D5DC6459DD1909C70188084201E1F) >
                        ERRORval = tline[whereSID + rightqSID + 2 :]
                    else:
                        # [To handle the case when [SID: ....] does not exists, find the immediate right of [ERROR], and then check if '[0x' (zeroX) exist]:
                        # 2018-05-14 20:35:34.800 [HOST:env-93835laio2use1][SERVER:CastorServer][PID:84762][THR:140030753163008]
                        # [Metadata Server][Error][0x8004140B] Object with ID '44F9CBE411E857B600000080EF854C73' and type 4 (Metric)
                        # is not found in metadata. It may have been deleted.
                        ERRORval = tline[rightqERROR + 1:]
                else:
                    # if [OID: ....] exists, find the imediate rightqt of OID.  Extract error string all the way to the end of the line
                    ERRORval = tline[rightqOID + 1:]

                # Exceptional rule to take care of very long text led by <rw_manipulations dumpdf_err
                whereMANIPULATION = ERRORval.find('<rw_manipulations')
                if (whereMANIPULATION > -1):
                    EXTRAval = ERRORval[whereMANIPULATION:whereMANIPULATION + 100] + ' ...'
                    ERRORval = ERRORval[1:whereMANIPULATION - 1]

                # Exceptional rule to take care of Big Data team log dump defect
                whereCSV = ERRORval.find('[SimpleCSVParser]')
                if (whereCSV > -1):
                    leftqDEXTRA = ERRORval[18 :].find('[')
                    if leftqDEXTRA == -1:
                        leftqDEXTRA = ERRORval[18 :].find('PROBLEM DESCRIPTION')
                    if leftqDEXTRA == -1:
                        leftqDEXTRA = ERRORval[18 :].find('LATEST STATUS SUMMARY')
                    if leftqDEXTRA == -1:
                        leftqDEXTRA = ERRORval[18 :].find('Cannot parse out numeric value from')+ 36
                    if leftqDEXTRA == -1:
                        EXTRAval = ''
                    else:
                        EXTRAval = '[SimpleCSVParser]: ' + ERRORval[leftqDEXTRA + 17 : ]
                        ERRORval = ERRORval[1 : (leftqDEXTRA + 17 -1)]

                # remove [TAB] and [NEW LINE] character
                ERRORval = ERRORval.replace("\t", " ")
                ERRORval = ERRORval.replace("\n", "")
                isMainLine = True
                nexttoMainline = True
            else:
                nexttoMainline = False
            currentLine = currentLine + 1
            tline = fin.readline()

        else:
            # if datetime string does not exist, check if there is an extra information
            # check if the previous line is a main[ERROR] line(ie.contains [Error] or {Fatal]
            EXTRAtmp = ''
            if nexttoMainline :

                while (tline[0:20].find(date_str) == -1) and ( not(tline == '')):
                    tline = tline.replace("\n", " ")
                    EXTRAtmp = EXTRAtmp + tline
                    tline = fin.readline()
                    currentLine = currentLine + 1
                EXTRAtmp = EXTRAtmp.replace("\t", " ")
                EXTRAtmp = EXTRAtmp.replace("\n", "")

                # Check if there is any term matched to the phrases from the truncation list which might be the XML or SQL dump and needs to be truncated
                if any(word in EXTRAtmp for word in truncat):
                    # Roll extra error back to the error except the following conditions
                    if ERRORval.find('[SimpleCSVParser]') > -1:
                        ERRORval = ERRORval + EXTRAtmp
                    else:
                        whereERRORTYPE = EXTRAtmp.find('Error type:')
                        if whereERRORTYPE > -1:
                            tempStr = EXTRAtmp[whereERRORTYPE + 12 : ]
                            leftqTYPE1 = [x for x in range(len(tempStr)) if tempStr.startswith('[', x)]
                            leftqTYPE2 = [x for x in range(len(tempStr)) if tempStr.startswith('(', x)]
                            leftqTYPE3 = [x for x in range(len(tempStr)) if tempStr.startswith('.', x)]
                            if leftqTYPE1 == []:
                                leftqTYPE1 = [99999, 99999]
                            if leftqTYPE2 == []:
                                leftqTYPE2 = [99999, 99999]
                            if leftqTYPE3 == []:
                                leftqTYPE3 = [99999, 99999]
                            leftqTYPE = min(leftqTYPE1[0], leftqTYPE2[0], leftqTYPE3[0])
                            extra0 = EXTRAtmp[whereERRORTYPE : whereERRORTYPE + leftqTYPE + 10]
                            extra1 = EXTRAtmp[whereERRORTYPE + whereERRORTYPE + 12 :]
                            EXTRAval = EXTRAval + ' ' + extra1
                            ERRORval = ERRORval + ' { ' + extra0 + ' }'
                        else:
                            EXTRAval = EXTRAval + ' { ' + EXTRAtmp + ' }'
                else:
                    ERRORval = ERRORval + ' {' + EXTRAtmp + '}'
                nexttoMainline = True
                isMainLine = True
            else:
                # it is NOT an legit message, so the line counter advanves by 1
                nexttoMainline = False
                tline = fin.readline()
                currentLine = currentLine + 1

        # Put all attribute values together to form a vector
        if (isMainLine or nexttoMainline):
            # Keyword Search: Search keyword on both ERROR and EXTRA columns. If matched, write keyword to the "KEYWORD" column
            max_kwd_weight = 0
            this_kwd_weight = 0
            keywordFound = False
            KWDval = ''
            which_keyword = 0

            # Keuword search against the keyword watch list
            while (not keywordFound) and (which_keyword < nb_keyword):
                loc_keyword = ERRORval.upper().find(df_keyword['ERROR KEYWORD'][which_keyword])
                if loc_keyword > -1:
                    this_kwd_weight = df_keyword['WEIGHT'][which_keyword]
                    if this_kwd_weight > max_kwd_weight:
                        max_kwd_weight = this_kwd_weight
                        KWDval = df_keyword['ERROR KEYWORD'][which_keyword]
                else:
                    if EXTRAval != '':
                        loc_keyword = EXTRAval.upper().find(df_keyword['ERROR KEYWORD'][which_keyword])
                        if loc_keyword > -1:
                            this_kwd_weight = df_keyword['WEIGHT'][which_keyword]
                            if this_kwd_weight > max_kwd_weight:
                                max_kwd_weight = this_kwd_weight
                                KWDval = df_keyword['ERROR KEYWORD'][which_keyword]
                which_keyword = which_keyword + 1

            # FATAL vs. ERROR check
            if (whereFATAL > -1):
                isFatal = 1
            else:
                isFatal = 0

            #  x0: BLITZ TEST
            #  x1: PID
            #  x2: TIME STAMP
            #  x3: PRODUCT
            #  x4: ERROR
            #  x5: IS FATAL
            #  x6: EXTRA
            #  x7: UID
            #  x8: SID
            #  x9: OID
            #  x10: THU
            #  x11: KEYWARD
            #  x12: WEIGHT (select the highest if many)
            #  x13: IDENTICALS (initialized by value 1)

            if len(EXTRAval)>256:
                EXTRAval = EXTRAval[0:255]

            errorVec = [value_list[2], PIDval, TIMESTAMPval, PRODUCTval, ERRORval, isFatal, EXTRAval, UIDval, SIDval, OIDval, THRval, KWDval, max_kwd_weight, 1]

            # Append the error vector into error array
            ary_err.append(errorVec)

    fin.close()

    # Convert the data type series to dataframe
    df_err = pd.DataFrame(ary_err, columns=columns)

    # Sort Keyword weight in decending order so the higher weight errors can be retained after finding unique single
    df_temp = df_err.sort_values(['WEIGHT'], ascending=False)
    # Use unique operation to remove duplicates, where the variables of:
    # uniq_err = the unique error text. loc1 = the index of the chosen unique. counts = duplicate counts
    uniq_err, loc1, counts = np.unique(df_temp['ERROR'], return_index=True, return_counts=True)

    # q is the number of unique errors
    q = np.size(uniq_err)

    # Assemble the entire row of chosen unique
    df_uniq_err = df_temp.iloc[loc1]

    # Add unique count column into the data frame
    df_uniq_err['IDENTICALS'] = counts.tolist()
    blz.toc()

    # Out unique error dataframe to .csv
    # out_filename1 = outfilename[0:-4]+'_dataframe.csv'
    out_filename2 = outfilename[0:-4] + '_parse.csv'
    print('Exporting '+ os.getcwd() + '\\' + out_filename2 )
    df_uniq_err.to_csv(out_filename2, index=False)
    print('\x1b[1;33m' + 'Done with [Parsing].' + '\x1b[0m')

############################
def predict():
    global df_uniq_err
    global tst_out
    global prediction
    global out_filename1
    global vec_filename
    global pty_filename

    print('\nStarting ' + '\x1b[6;30;42m' + ' STEP 4 ' + '\x1b[0m')

    # nb_error is the number of unique errors
    nb_error = len(df_uniq_err['ERROR'])
    tst_clean = [None] * nb_error

    # Load ML Training model
    print("******  Vectorization gallery ... ")
    df_gallery = pd.read_csv(pty_filename, encoding="ISO-8859-1", engine='python')
    vectorizer2 = pickle.load(open(vec_filename, "rb"), encoding='iso-8859-1')

    x = vectorizer2.transform(df_gallery['ERROR TOKENS'])
    X = x.toarray()
    nb_gallery = len(X)

    print("****** Text Regular Expression: probe  ... ")
    blz.tic()
    # text regular expression operation
    regex_pat123 = re.compile(r'[^a-zA-Z0-9\s]', flags=re.IGNORECASE)
    regex_pat = re.compile(r'[^a-zA-Z\s]', flags=re.IGNORECASE)

    for k in range(nb_error):
        if ( k % 200 ==1 ) or (k+200 >= nb_error):
            sg.OneLineProgressMeter('Vectorization', k+1, nb_error, 'key')
        temp = df_uniq_err['ERROR'].iloc[k]
        temp = temp.replace("/", " ")
        temp = temp.replace("_", " ")
        temp = temp.replace("-", " ")
        temp = temp.replace("=", " ")
        temp = temp.replace(";", " ")
        temp = temp.replace(".", " ")
        temp = temp.replace("'", "")
        # take care remove nonprintable characters
        # temp = temp.replace("\xc3",'')
        # temp = temp.replace("\xa4",'')
        # temp = temp.replace("\xe5",'')
        temp = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', temp)
        #
        # tst_words = nltk.word_tokenize(temp)
        # tst_series = pd.Series(tst_words)
        tst_series = pd.Series(temp)
        # keep only words
        tst_clean1 = tst_series.str.replace(regex_pat123, ' ')
        mask = ((tst_series.str.len() == 32) | (tst_series.str.len() == 33)) & (~tst_series.str.islower())
        tst_clean1.loc[pd.Series.as_matrix(mask)] = 'GUID'
        tst_clean1 = tst_series.str.replace(regex_pat, ' ')  # join the cleaned words in a list
        tst_clean2 = tst_clean1.str.cat(sep=' ')
        tst_clean[k] = tst_clean2
    blz.toc()

    print("****** Cluster Prediction on testing samples ... ")
    blz.tic()

    print('Prediction: [learning Gallery: ' + pty_filename + '] [vector space: ' + vec_filename + ']')
    y = vectorizer2.transform(tst_clean)
    Y = y.toarray()
    # prediction = loaded_model.predict(Y)

    bestMatch_index = [0] * nb_error
    bestMatch_score = [0] * nb_error
    prediction = [0] * nb_error
    pred_str = [None] * nb_error
    frequency = [0] * nb_error

    for i in range(nb_error):
        # Pop up a progress bar
        if ( i % 200 ==1 ) or (i+200 >= nb_error):
            sg.OneLineProgressMeter('Simaility Matching', i+1, nb_error, 'key')

        temp_similarity= [0] * nb_gallery
        for j in range(nb_gallery):
            temp_similarity[j] = 1 - distance.cosine( Y[i], X[j])

        bestMatch_index[i] = np.argmax(temp_similarity)
        bestMatch_score[i] = max(temp_similarity)
        prediction[i] = df_gallery.iloc[bestMatch_index[i]]['CLUSTER TAG']
        frequency[i] = df_gallery.iloc[bestMatch_index[i]]['FREQUENCY']

        # Assign the cluster tags
        pred_str[i] = ['C' + str(prediction[i])]

    vec_filename = outfilename[0:-4] + '_vsp.npz'
    np.savez(vec_filename, Y)
    print('Saving vector space projection as .npz file for internal algorithm evaluation:  ' + os.getcwd() + '\\' + vec_filename)
    blz.toc()

    # Put testing outputs together in a data frame format
    print("Generating Error Ranking Output ... ")
    blz.tic()

    tst_uniq_df = df_uniq_err

    # MATCH SCORE
    df_sim = pd.DataFrame(bestMatch_score, columns=["SIMILARITY SCORE"])


    # CLUSTER_ID
    tst_df1 = pd.DataFrame(pred_str, columns=["CLUSTER_ID"])

    # NO. OF ALIKE
    tst_d2 = np.bincount(prediction)
    tst_d2 = tst_d2[prediction]
    tst_df2 = pd.DataFrame(tst_d2, columns=["NO. OF ALIKE"])

    # MODEL FREQUENCY (MODEL CLUSTER SIZE)


    # tst_d3 = frequency.astype('float', copy=True) / sum(frequency)
    tst_d3 = frequency / sum(frequency)
    tst_df3 = pd.DataFrame(frequency, columns=["MODEL CLUSTER SIZE"])

    # MATCH SCORE
    # tst_d4 = bestMatch_score
    # tst_df4 = pd.DataFrame(tst_d4, columns=["MATCH SCORE"])

    # ERROR KPI
    # prob. of test in-cluster
    tst_d2p = tst_d2.astype('float', copy=True) / nb_error
    # prob. of gallery in-cluster
    tst_d3p = tst_d3.astype('float', copy=True) / sum(tst_d3)

    tst_d5 = abs(np.log(tst_d3p * tst_d2p))
    tst_df5 = pd.DataFrame(tst_d5, columns=["ERROR KPI"])

    # BOOSTED ERROR KPI
    tst_d6 = np.asarray((1 + tst_uniq_df['WEIGHT'])*tst_d5)
    tst_df6 = pd.DataFrame(tst_d6, columns=["BOOSTED ERROR KPI"])

    # Ignore the dataframe index before concatenation
    tst_uniq_df.reset_index(drop=True, inplace=True)

    # Concate with additional atributes
    # tst_out = pd.concat([tst_uniq_df, df_sim, tst_df1, tst_df2, tst_df3, tst_df5, tst_df6], axis=1)
    tst_out = pd.concat([tst_uniq_df, df_sim, tst_df1, tst_df2, tst_df3, tst_df5, tst_df6], axis=1)

    out_filename1 = outfilename[0:-4]+'_datacube_nopa.csv'
    # out_filename1 = 'out_DSSErr' + tst_filename_prefix + 'n' + str(node) + '_' + env_id + office + '_datacube.csv'
    tst_out.to_csv(out_filename1, index=False)
    print('exporting ' + os.getcwd() + '\\' + out_filename1)
    blz.toc()
    print('\x1b[1;33m' + "Done with [Ranking]." + '\x1b[0m')


############################
def paTimeline():
    global envStr
    global envfilename
    global pafilename
    global tst_out
    # global prediction
    global df_err_pa
    global outfilename
    global out_filename0
    global df_final

    print('\nStarting ' + '\x1b[6;30;42m' + ' STEP 6 '+ '\x1b[0m')
    FMT1 = '%Y-%m-%d %H:%M:%S'

    df_pa0 = pd.read_csv(pafilename)
    df_ins = pd.read_csv(envfilename)

    pa_ools =list(itemgetter(2, 3, 4)(list(df_pa0.head(0))))
    instance_id = df_ins[df_ins['instance_name'] == envStr]['instance_id'].item()
    print('Instance ID: ', instance_id)

    df_pa = df_pa0[df_pa0['instance_id'] == instance_id ][pa_ools]
    df_pa = df_pa.rename({'metric_retrieve_time':'ROUNDED TIME STAMP', 'cpu_avg_usage':'CPU', 'mem_avg_usage':'MEMORY'}, axis='columns')

    pa_timestamp = pd.to_datetime(df_pa['ROUNDED TIME STAMP'])

    # t1: Rounded TimeStamp lower bound (Error TS lowbound - 2 hours) ; t2: Rounded TimeStamp upper bound (Error TS lowbound + 2 hours)
    # ETC to UTC (+5 hours) and +/- 2 hours for PA range
    tst_out2 = tst_out
    # update TIME STAMP from ETS to UTC time zone
    tst_out2['TIME STAMP'] = tst_out2['TIME STAMP'] + pd.DateOffset(hours=5)
    error_ts_lowbound = min(tst_out2['TIME STAMP'])
    error_ts_upperbound = max(tst_out2['TIME STAMP'])
    t1 = error_ts_lowbound  - pd.DateOffset(hours=2)
    t2 = error_ts_upperbound + pd.DateOffset(hours=2)
    df_pa2 = df_pa[pa_timestamp.between(t1, t2)]

    # PA time sample rate (resolution) is 5 min.
    detla_t = 5

    print('****** Aligning Error timestamp and PA rounded timestamp [down-sampling to ' + str(detla_t)+ ' min interval] ...')
    blz.tic()
    pa_time = pd.to_datetime(df_pa2['ROUNDED TIME STAMP'])
    err_time = tst_out2['TIME STAMP']

    nb_error = len(tst_out2)
    time_allign_idx = [None]*nb_error
    old_time = [None]*nb_error
    round_time = [None]*nb_error
    err_local_clust = list([0] * nb_error)
    count_in_time = list([0] * nb_error)
    norm_cpu = list([0] * nb_error)
    norm_memory = list([0] * nb_error)


    for i in range(nb_error):
        err_time = tst_out2.iloc[i]['TIME STAMP']
        old_time[i] = err_time
        # find the time differences between the error outlier timestamp vs all PA timestamps
        time_delta = err_time - pd.to_datetime(pa_time)
        idx = np.argmin(abs(np.array(time_delta)))  # find the closest PA timestamp, temp is the PA index

        if not (idx == 0 and (err_time < pa_time.iloc[0] - timedelta(minutes=detla_t) or err_time > pa_time.iloc[-1] + timedelta(minutes=detla_t))):
            # if aligned timestamp is far below the lower bound (< t-5 mins) or beyond the upper bound (> t+5 mins.), do not assign corresponding rounded timestamp
            time_allign_idx[i] = idx
        else:
            time_allign_idx[i] = 0

        round_time[i] = pa_time.iloc[idx]
    blz.toc()

    print("\nArranging dataframe output  ...")
    blz.tic()
    #  x0: BLITZ TEST
    #  x1: PID
    #  x2: TIME STAMP
    #  x3: PRODUCT
    #  x4: ERROR
    #  x5: IS FATAL
    #  x6: EXTRA
    #  x7: UID
    #  x8: SID
    #  x9: OID
    #  x10: THU
    #  x11: KEYWARD
    #  x12: WEIGHT (select the highest if many)
    #  x13: IDENTICALS (initialized by value 1)
    #  x14: CLUSTER_ID
    #  x15: MODEL CLUSTER SIZE
    #  x16: NO.OF ALIKE
    #  x17: ERROR KPI
    #  x18: BOOSTED ERROR KPI
    #  x19: ROUNDED TIME STAMP
    #  x20: CPU
    #  x21: MEMORY
    #  x22: COUNTS OF SIMILAR ERRORS FROM THE SAME CLUSTER
    #  x23: POPULATION NORMALIZED KPI
    #  x24: ERROR COUNT PER TIMESTAMP
    #  x25: POPULATION NORMALIZED CPU
    #  x26: POPULATION NORMALIZED MEMORY

    # Re-index all error outlier samples
    df_tstout = pd.DataFrame(tst_out2, index=range(nb_error))

    # add rounded timestamps and reassign index
    df_x19 = pd.DataFrame(round_time, columns=['ROUNDED TIME STAMP'], index=range(nb_error))
    # Use the aligned time index to extract PA performance
    df_x20 = df_pa2.iloc[time_allign_idx][['CPU', 'MEMORY']]
    # Drop the index from error outlier data frame
    df_x21 = df_x20.reset_index(drop=True)
    #
    time_uniq, loc4, counts4 = np.unique(df_x19, return_index=True, return_counts=True)
    nb_ts = len(time_uniq)

    #
    for i in range(nb_ts):
        select_indices = list(np.where(df_x19 == time_uniq[i])[0])
        time_block = prediction[select_indices]
        # Number of samples in the time block
        nb_tb = len(time_block)

        # find unique cluster tags and counts within a time block
        clust_uniq, loc5, counts5 = np.unique(time_block, return_index=True, return_counts=True)
        for j in range(nb_tb):
            count_in_time[select_indices[j]] = nb_tb
            norm_cpu[select_indices[j]] = df_x20.iloc[select_indices[0]]['CPU'] / nb_tb
            norm_memory[select_indices[j]] = df_x21.iloc[select_indices[0]]['MEMORY'] / nb_tb
            #
            select_index = np.where(clust_uniq == time_block[j])[0][0]
            err_local_clust[select_indices[j]] = counts5[select_index]
    df_x22 = pd.DataFrame(err_local_clust, columns= ["SIMILAR ERROR COUNTS IN CLUSTER"])
    df_x23 = pd.DataFrame(tst_out2["ERROR KPI"].as_matrix(columns=None) /err_local_clust/count_in_time, columns=["NORMALIZED KPI"])

    df_x24 = pd.DataFrame({'ERROR COUNT PER TIMESTAMP': count_in_time})
    df_x25 = pd.DataFrame({'NORMALIZED CPU': norm_cpu})
    df_x26 = pd.DataFrame({'NORMALIZED MEMORY': norm_memory})

    df_err2 = pd.concat([df_tstout, df_x19, df_x21, df_x22, df_x23, df_x24, df_x25, df_x26], axis=1)
    # patching the CPU and Memory values outside TimeStamp with Errors

    df_final = df_pa2.append(df_err2)
    df_final = df_final.sort_values(by='BOOSTED ERROR KPI', ascending=False)
    df_final = df_final.reset_index(drop=True)

    df_length =  len(df_final)
    for i in range(df_length):
        if np.isnan(df_final.iloc[i]['NORMALIZED CPU']):
            if np.isnan(df_final.iloc[i]['ERROR COUNT PER TIMESTAMP']):
                df_final['NORMALIZED CPU'][i] = df_final['CPU'][i]
                df_final['NORMALIZED MEMORY'][i] = df_final['MEMORY'][i]
    blz.toc()
    # to remove the rows without errors within time interval
    ss = np.array(pd.to_datetime(df_final['ROUNDED TIME STAMP']) <= error_ts_upperbound)
    tt = np.array(pd.to_datetime(df_final['ROUNDED TIME STAMP']) >= error_ts_lowbound)
    uu = np.array(np.isnan(df_final['ERROR COUNT PER TIMESTAMP']))

    df_final = df_final.drop(df_final[ss & tt & uu].index)
    df_final = df_final.reset_index(drop=True)

    # OUTPUT RESULT AS .CSV FOR DATA CUBE
    blz.tic()
    print("Saving testing results ... ")
    out_filename0 = outfilename[0:-4] + '_datacube_pa.csv'
    print('Exporting '+ os.getcwd() + ' ' + out_filename0)
    df_final.to_csv(out_filename0, index=False)
    blz.toc()
    print('\x1b[1;33m' + "Done with [Time Stamp Correlation]." + '\x1b[0m')

############################
def push2cube_nopa():

    global value_list
    global tst_out
    global out_filename1
    global isLDAP
    global this_prefix

    FMT1 = '%Y-%m-%d %H:%M:%S'

    print('\nStarting ' + '\x1b[6;30;42m' + 'PUSH DATAFRAME TO MSTR CUBE (WO. PA): ' + '\x1b[0m')
    blz.tic()

    df_cube = pd.read_csv(out_filename1)
    df_cube['PID'] = df_cube['PID'].apply(str)

    # datasetName = this_prefix + 'err_n' + value_list[6] + '_nopa'
    # tableName = this_prefix + 'ErrorRank_n' + value_list[6] + '_nopa'
    # cubeinfo_name = 'Cube Info_' + this_prefix + 'n' + value_list[6] + '_nopa.txt'
    # datasetName0 = this_prefix + 'cube' + value_list[3] + '_n' + value_list[6] + '_nopa'

    datasetName = this_prefix + 'err_nopa'
    tableName = this_prefix + 'ErrorRank_nopa'
    cubeinfo_name = 'Cube Info_' + this_prefix + '_nopa.txt'
    datasetName0 = this_prefix + 'cube' + value_list[3] + '_nopa'


    # Authentication request and connect to the Rally Analytics project
    # is LDAP login (1) or standard user (0)
    if isLDAP == 1:
        conn = microstrategy.Connection(base_url=baseURL, login_mode=16,username=value_list[0], password=value_list[1], project_name=projName)
    else:
        conn = microstrategy.Connection(base_url=baseURL, username=value_list[0], password=value_list[1], project_name=projName)
    conn.connect()

    print("Connect to " + baseURL)

    # Create a new cube or use the existing cube
    if var1.get() == 1:
        # if the cube does not exist, acquire Data Set Id & Table Id, and create a new cube
        newDatasetId, newTableId = conn.create_dataset(data_frame=df_cube, dataset_name=datasetName, table_name=tableName)
        # Store Data Set Id and Table Id locally
        cubeInfoFile = open(cubeinfo_name, 'w')
        cubeInfoFile.write(newDatasetId + '\n')
        cubeInfoFile.write(newTableId)
        cubeInfoFile.close()
        print("CREATE Cube on URL: " + baseURL[:-25])
        print('[ Dataset Name: ' + datasetName + ' \ Cube ID = ' + newDatasetId + ']   [Table Name: ' + tableName + ' \ Table ID = ' + newTableId + ' ]')
    else:
        # Read saved cube IDs

        cubeInfoFile = open(cubeinfo_name, 'r')
        datasetID = cubeInfoFile.read().splitlines()
        cubeInfoFile.close()
        # Establish cube connection
        conn.update_dataset(data_frame=df_cube, dataset_id=datasetID[0], table_name=tableName, update_policy='add')
        print("UPDATE Cube on URL: " + baseURL[:-25])
        print("Dataset Name " + datasetName + "[Cube ID: " + datasetID[0] + "   Table Name: " + tableName + "]")

    print("CREATE a backup cube: " + datasetName0)
    newDatasetId0, newTableId0 = conn.create_dataset(data_frame=df_cube, dataset_name=datasetName0,table_name=tableName)
    blz.toc()
    print('\x1b[1;33m' + "Done with [Output to MSTR Cube for Dossier Reporting (without PA)]" + '\x1b[0m')

############################
def push2cube_pa():

    # global value_list
    global df_final
    global baseURL
    global projName
    global out_filename0

    print('\nStarting ' + '\x1b[6;30;42m' + 'PUSH DATAFRAME TO MSTR CUBE (W. PA): ' + '\x1b[0m')
    blz.tic()
    df_cube = pd.read_csv(out_filename0)
    df_cube['PID'] = df_cube['PID'].apply(str)

    datasetName = 'DemoTest_n' + value_list[6] + '_pa'
    tableName = 'ErrorRank_demo_n' + value_list[6] + '_pa'
    cubeinfoName = 'demoInfo_n' + value_list[6] + '_pa.txt'
    datasetName0 = 'DemoTest_' + value_list[3] + '_n' + value_list[6] + '_pa'

    isNewCube = False
    if value_list[2] == '':
        isNewCube = True

    # Authentication request and connect to the Rally Analytics project
    conn = microstrategy.Connection(
        base_url=baseURL, login_mode = 16, username=value_list[0], password=value_list[1], project_name=projName)
    conn.connect()

    print("Connect to " + baseURL)

    if var1.get() == 1:
        # if the cube does not exist, acquire Data Set Id & Table Id, and create a new cube
        newDatasetId, newTableId = conn.create_dataset(data_frame = df_cube, dataset_name = datasetName, table_name = tableName)

        # Store Data Set Id and Table Id locally
        cubeInfoFile = open(cubeinfoName, 'w')
        cubeInfoFile.write(newDatasetId + '\n')
        cubeInfoFile.write(newTableId)
        cubeInfoFile.close()
        print("CREATE Cube on URL: " + baseURL[:-25])
        print('[ Dataset Name: ' + datasetName + ' \ Cube ID = ' + newDatasetId + ']   [Table Name: ' + tableName + ' \ Table ID = ' + newTableId + ' ]')
    else:
        # Read saved cube IDs
        cubeInfoFile = open(cubeinfoName, 'r')
        datasetID = cubeInfoFile.read().splitlines()
        cubeInfoFile.close()
        # Establish cube connection
        conn.update_dataset(data_frame=df_cube, dataset_id = datasetID[0], table_name = tableName, update_policy='add')
        print("UPDATE Cube on URL: " + baseURL[:-25])
        print("Dataset Name " + datasetName + "[Cube ID: " + datasetID[0] + "   Table Name: " + tableName + "]")

    print("CREATE a backup cube: " + datasetName0)
    newDatasetId0, newTableId0 = conn.create_dataset(data_frame=df_cube, dataset_name=datasetName0, table_name=tableName)
    blz.toc()
    print('\x1b[1;33m' + "Done with [Output to MSTR Cube for Dossier Reporting (with PA)]" + '\x1b[0m')

##############################################################
if __name__ == '__main__':

    try:
        root.destroy()
    except:
        pass
    root = Tk()
    root.geometry('560x960')
    root.title('MSTR Error Log Analytic Tool')

    row0 = 12



    l1 = Label(root, text="MicroStrategy Web: ...")
    l1.grid(row= row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)
    #
    var2 = IntVar()
    var2.set(0)
    c2 = tk.Checkbutton(root, variable=var2, onvalue=1, offvalue=0, text="LDAP Login")
    c2.grid(row=row0, column=1, columnspan=2, sticky="w", padx=6, pady=12)
    row0 = row0 + 1
    #
    var0 = StringVar(root)
    p0 = OptionMenu(root, var0, *baseURLs)
    p0.config(bg="LIGHT BLUE")
    p0.grid(row= row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1
    #
    ents = makeform(root, fields)
    var1 = IntVar()
    var1.set(0)
    c1 = tk.Checkbutton(root, variable=var1, onvalue=1, offvalue=0, text="Create a New Cube?    Cube Prefix:")
    c1.grid(row=row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)

    entry3 = tk.Entry(state='normal')
    entry3.config(bg="LIGHT YELLOW")
    entry3.grid(row=row0, column=1, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1

    f = ttk.Separator(root)
    f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1

    b1 = tk.Button(root, text='STEP 1: Click to Confirm All Setting ...',  command=(lambda e=ents: fetch(e)), highlightbackground = "ivory2", highlightthickness=2)
    b1.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1

    # f = ttk.Separator(root)
    # f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    # row0 = row0 + 1

    b2 = tk.Button(root, text='STEP 2: Select an Input DSSError.log File for Analyzing ...', command = logSelect, highlightbackground = "light blue", highlightthickness=2)
    outfilename2 = tk.Button(b2)
    b2.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
    row0 = row0 + 1

    # f = ttk.Separator(root)
    # f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    # row0 = row0 + 1

    b3 = tk.Button(root, text='STEP 3: Proceed Text Cleaning and Parsing ...',  command = parse, highlightbackground = "papaya whip", highlightthickness=2)
    b3.grid(row=row0, column=0, columnspan=2,sticky="ew",  padx=6, pady=6)
    row0 = row0 + 1

    # f = ttk.Separator(root)
    # f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    # row0 = row0 + 1

    b4a = tk.Button(root, text='*Select Text Vectorizer (.sav)',  command = vecSelect, highlightbackground = "light blue", highlightthickness=2)
    b4a.grid(row=row0, column=0, columnspan=1,sticky="ew",  padx=6, pady=6)

    b4b = tk.Button(root, text='*Select Cluster Prototypes (.csv)',  command = gallerySelect, highlightbackground = "light blue", highlightthickness=2)
    b4b.grid(row=row0, column=1, columnspan=1,sticky="ew",  padx=6, pady=6)
    row0 = row0 + 1

    b4 = tk.Button(root, text='STEP 4: Error Clustering and Ranking ... ',  command = predict, highlightbackground = "peach puff", highlightthickness=2)
    b4.grid(row=row0, column=0, columnspan=2,sticky="ew",  padx=6, pady=6)
    row0 = row0 + 1

    # f = ttk.Separator(root)
    # f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    # row0 = row0 + 1

    b5 = tk.Button(root, text='Output to MSTR Cube for Dossier Reporting (without PA) ... ', command=push2cube_nopa, highlightbackground = "light pink", highlightthickness=2)
    b5.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
    row0 = row0 + 1

    f = ttk.Separator(root)
    f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1

    l0 = Label(root, text="DO NOT CONTINUE IF YOU DO NOT HAVE PA DATA", fg = "red")
    l0.grid(row= row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)
    row0 = row0 + 1

    b6 = tk.Button(root, text='STEP 5: Select an Input PA Data File to Correlate Detected Errors ...', command = paSelect, highlightbackground = "PaleGreen1", highlightthickness=2)
    pafilename = tk.Button(b5)
    b6.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
    row0 = row0 + 1

    # f = ttk.Separator(root)
    # f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    # row0 = row0 + 1

    b7 = tk.Button(root, text='STEP 6: Error Correlation vs. CPU/MEMORY Utilization with TimeStamp ... ', command = paTimeline, highlightbackground = "PaleGreen2", highlightthickness=2)
    b7.grid(row=row0, column=0, columnspan=2,sticky="ew",  padx=6, pady=6)
    row0 = row0 + 1

    # f = ttk.Separator(root)
    # f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    # row0 = row0 + 1

    b8 = tk.Button(root, text='Output to MSTR Cube for Dossier Reporting (with PA) ... ', command = push2cube_pa, highlightbackground = "PaleGreen3", highlightthickness=2)
    b8.grid(row=row0, column=0, columnspan=2,sticky="ew",  padx=6, pady=6)
    row0 = row0 + 1

    # f = ttk.Separator(root)
    # f.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    # row0 = row0 + 1

    b9 = tk.Button(root, text='Done ...', command= close_window, highlightbackground ="ivory2", highlightthickness=2)
    b9.grid(row=row0, column=0, columnspan=2,sticky="ew",  padx=6, pady=6)
    row0 = row0 + 1

    l0 = Label(root, text="ERROR LOG ANALYTICS")
    l0.config(font=("Courier", 32))
    l0.grid(row= row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)

    root.mainloop()



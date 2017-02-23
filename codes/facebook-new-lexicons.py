# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:57:42 2016

@author: TasosLytos
"""

import urllib2
import json
import datetime
import csv
import time
#import mysql.connector
import pymysql.cursors
#import unicodedata
#from pandas import DataFrame, Series
import pandas as pd
import subprocess
import re
import nltk
#import fileinput
#import numpy as np

# boolean debug variable
debug_variable = False 

# temporary
app_id = "<FILL IN>"
app_secret = "<FILL IN>" 
page_name = "acer"

# permanent
access_token = "********************" 

# connect to DataBase
def connectDb():
    connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
						  db='')
    return connection

# finds a specific word in a given text 
def wordInText(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

# finds the number of each POS the status has 
def firstProcess():
  
    cc_counter = 0
    cd_counter = 0
    dt_counter = 0
    ex_counter = 0
    fw_counter = 0
    in_counter = 0
    jj_counter = 0
    jjr_counter = 0
    jjs_counter = 0
    ls_counter = 0
    md_counter = 0
    nn_counter = 0
    nns_counter = 0
    nnp_counter = 0
    nnps_counter = 0
    pdt_counter = 0
    pos_counter = 0
    prp_counter = 0
    prp6_counter = 0
    rb_counter = 0
    rbr_counter = 0
    rbs_counter = 0
    rp_counter = 0
    sym_counter = 0
    to_counter = 0
    uh_counter = 0
    vb_counter = 0
    vbd_counter = 0
    vbg_counter = 0
    vbn_counter = 0
    vbp_counter = 0
    vbz_counter = 0
    wdt_counter = 0    
    wp_counter = 0
    wp6_counter = 0
    wrb_counter = 0

    file=open("POSTagger/processed_output_POS.txt","r+")
    
    for word in file.read().split():    
        if wordInText("_cc",word):
            cc_counter = cc_counter + 1
        if wordInText("_cd",word):
            cd_counter = cd_counter + 1
        if wordInText("_dt",word):
            dt_counter = dt_counter + 1
        if wordInText("_ex",word):
            ex_counter = ex_counter + 1
        if wordInText("_fw",word):
            fw_counter = fw_counter + 1
        if wordInText("_in",word):
            in_counter = in_counter + 1
        if wordInText("_jj",word):
            jj_counter = jj_counter + 1
        if wordInText("_jjr",word):
            jjr_counter = jjr_counter + 1
        if wordInText("_jjs",word):
            jjs_counter = jjs_counter + 1
        if wordInText("_ls",word):
            ls_counter = ls_counter + 1
        if wordInText("_md",word):
            md_counter = md_counter + 1
        if wordInText("_nn",word):
            nn_counter = nn_counter + 1
        if wordInText("_nns",word):
            nns_counter = nns_counter + 1
        if wordInText("_nnp",word):
            nnp_counter = nnp_counter + 1
        if wordInText("_nnps",word):
            nnps_counter = nnps_counter + 1
        if wordInText("_pdt",word):
            pdt_counter = pdt_counter + 1
        if wordInText("_pos",word):
            pos_counter = pos_counter + 1
        if wordInText("_prp",word):
            prp_counter = prp_counter + 1
        if wordInText("_prp6",word):
            prp6_counter = prp6_counter + 1
        if wordInText("_rb",word):
            rb_counter = rb_counter + 1
        if wordInText("_rbr",word):
            rbr_counter = rbr_counter + 1
        if wordInText("_rbs",word):
            rbs_counter = rbs_counter + 1	
        if wordInText("_rp",word):
            rp_counter = rp_counter + 1
        if wordInText("_sym",word):
            sym_counter = sym_counter + 1
        if wordInText("_to",word):
            to_counter = to_counter + 1
        if wordInText("_uh",word):
            uh_counter = uh_counter + 1
        if wordInText("_vb",word):
            vb_counter = vb_counter + 1
        if wordInText("_vbd",word):
            vbd_counter = vbd_counter + 1
        if wordInText("_vbg",word):
            vbg_counter = vbg_counter + 1
        if wordInText("_vbn",word):
            vbn_counter = vbn_counter + 1
        if wordInText("_vbp",word):
            vbp_counter = vbp_counter + 1
        if wordInText("_vbz",word):
            vbz_counter = vbz_counter + 1
        if wordInText("_wdt",word):
            wdt_counter = wdt_counter + 1			
        if wordInText("_wp",word):
            wp_counter = wp_counter + 1
        if wordInText("_wp6",word):
            wp6_counter = wp6_counter + 1
        if wordInText("_wrb",word):
            wrb_counter = wrb_counter + 1	

    return cc_counter, cd_counter, dt_counter, ex_counter, fw_counter, in_counter, jj_counter, jjr_counter, jjs_counter, ls_counter, md_counter, nn_counter, nns_counter, nnp_counter, nnps_counter, pdt_counter, pos_counter, prp_counter, prp6_counter, rb_counter, rbr_counter, rbs_counter, rp_counter, sym_counter, to_counter, uh_counter, vb_counter, vbd_counter,  vbg_counter, vbn_counter, vbp_counter, vbz_counter, wdt_counter, wp_counter, wp6_counter, wrb_counter
	
    file.close();
	
# the function that makes persistent http request
def requestUntilSucceed(url):
    req = urllib2.Request(url)
    success = False
    while success is False:
        try: 
            response = urllib2.urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception, e:
            print e
            time.sleep(5)

            print "Error for URL %s: %s" % (url, datetime.datetime.now())
            print "Retrying."

    return response.read()

# needed to write tricky unicode correctly to csv
def unicodeNormalize(text):
    return text.translate({ 0x2018:0x27, 0x2019:0x27, 0x201C:0x22, 0x201D:0x22,
                            0xa0:0x20 }).encode('utf-8')

# returns the desired url, after the necessary construction
def getFacebookPageUrl(page_name, access_token, num_statuses):

    url_base = "https://graph.facebook.com/v2.6"
    node = "/%s/posts" % page_name
    url_fields = "/?fields=message,link,created_time,type,name,id," + \
            "comments.limit(0).summary(true),shares,reactions" + \
            ".limit(0).summary(true)"
    parameters = "&limit=%s&access_token=%s" % (num_statuses, access_token)
    url = url_base + node + url_fields + parameters

    # print the url in order to check it manually if I want
    if debug_variable == True:
		print url

    # retrieve data
    data = json.loads(requestUntilSucceed(url))

    return data

# returns the reactions 
# must ask for each reaction explicitly 	
def getReactions(status_id, access_token):

    url_base = "https://graph.facebook.com/v2.6"
    node = "/%s" % status_id
    reactions = "/?fields=" \
            "reactions.type(LIKE).limit(0).summary(total_count).as(like)" \
            ",reactions.type(LOVE).limit(0).summary(total_count).as(love)" \
            ",reactions.type(WOW).limit(0).summary(total_count).as(wow)" \
            ",reactions.type(HAHA).limit(0).summary(total_count).as(haha)" \
            ",reactions.type(SAD).limit(0).summary(total_count).as(sad)" \
            ",reactions.type(ANGRY).limit(0).summary(total_count).as(angry)"
    parameters = "&access_token=%s" % access_token
    url = url_base + node + reactions + parameters

    # retrieve data
    data = json.loads(requestUntilSucceed(url))

    return data

# checks if everything goes right
# some items does not exist necessarily
def processFacebookPageData(status, access_token):

    status_id = status['id']
    status_message = '' if 'message' not in status.keys() else \
            unicodeNormalize(status['message'])
    link_name = '' if 'name' not in status.keys() else \
            unicodeNormalize(status['name'])
    status_type = status['type']
    status_link = '' if 'link' not in status.keys() else \
            unicodeNormalize(status['link'])

    # the time is in UTC, I transform it in EEST 
	# I also change its format in order to be easier manipulate

    status_published = datetime.datetime.strptime(
            status['created_time'],'%Y-%m-%dT%H:%M:%S+0000')
    status_published = status_published + \
            datetime.timedelta(hours=+3) # EEST
    status_published = status_published.strftime(
            '%Y-%m-%d %H:%M:%S') 

    # the data we thrived from Facebook is in a python dictionary
	# the number of reactions/comments/shares are nested items
	# nested items require chaining dictionary keys 
	
    num_reactions = 0 if 'reactions' not in status else \
            status['reactions']['summary']['total_count']
    num_comments = 0 if 'comments' not in status else \
            status['comments']['summary']['total_count']
    num_shares = 0 if 'shares' not in status else status['shares']['count']

	# Facebook added reactions after 24 February 2016
	# So we must check the date of the status published 

    reactions = getReactions(status_id, access_token) if \
            status_published > '2016-02-24 00:00:00' else {}

    num_likes = 0 if 'like' not in reactions else \
            reactions['like']['summary']['total_count']

    # in case of pre-reaction status we set number of reactions equal to  
    # number of likes

    num_likes = num_reactions if status_published < '2016-02-24 00:00:00' \
            else num_likes

	# function that returns the number of each reaction 
	
    def getNumberTotalReactions(reaction_type, reactions):
        if reaction_type not in reactions:
            return 0
        else:
            return reactions[reaction_type]['summary']['total_count']

    num_loves = getNumberTotalReactions('love', reactions)
    num_wows = getNumberTotalReactions('wow', reactions)
    num_hahas = getNumberTotalReactions('haha', reactions)
    num_sads = getNumberTotalReactions('sad', reactions)
    num_angrys = getNumberTotalReactions('angry', reactions)

    # check the status that have been extracted from the facebook page 
	
    if debug_variable :
		with open('POSTagger\processed_output.txt', 'a') as fout:
			fout.write(status_message)

    # Return a tuple of all processed data

    return (status_id, status_message, link_name, status_type, status_link,
            status_published, num_reactions, num_comments, num_shares,
            num_likes, num_loves, num_wows, num_hahas, num_sads, num_angrys)

# store the information from the facebook page to my database
def storeFacebookInformationDataBase(page_name, access_token):
    
    scrape_starttime = datetime.datetime.now()
    print "Scraping %s Facebook Page, at time: %s\n" % (page_name, scrape_starttime)
    num_processed = 0   # keep a count on how many we've processed
    statuses = getFacebookPageUrl(page_name, access_token, 100)
    #while has_next_page:
    while num_processed<100:            
        for status in statuses['data']:

            # Ensure it is a status with the expected metadata
            if 'reactions' in status:
                if debug_variable :
                    print(processFacebookPageData(status,access_token))
                
                status_id_db, status_message_db, link_name_db, status_type_db, status_link_db, status_published_db, num_reactions_db, num_comments_db, num_shares_db, num_likes_db, num_loves_db, num_wows_db, num_hahas_db, num_sads_db, num_angrys_db  = processFacebookPageData(status,access_token)

            # output progress occasionally to make sure code is not
            # stalling
            num_processed += 1
            if num_processed % 100 == 0:
                print "%s Statuses Processed: %s" % \
                        (num_processed, datetime.datetime.now())
            page_data = status_id_db, status_message_db, link_name_db, status_type_db, status_link_db, status_published_db, num_reactions_db, num_comments_db, num_shares_db, num_likes_db, num_loves_db, num_wows_db, num_hahas_db, num_sads_db, num_angrys_db
            
            if debug_variable:
				print(page_data)
            
            # SQL statement for adding Facebook data
            insert_info = ("INSERT INTO statuses_pos " "(status_id, status_message, link_name, status_type, status_link, status_published, num_reactions, num_comments, num_shares, num_likes, num_loves, num_wows, num_hahas, num_sads, num_angrys)" "VALUES (%s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)") 

            connection = connectDb()
            cursor = connection.cursor()
            try:
                with connection.cursor() as cursor:
                    # Create a new record
                    # insert the data we pulled into db
                    cursor.execute(insert_info, page_data)
                    print("insert executions is done")

                # commit the data to the db
                connection.commit()
            except ValueError as error:
                print(error)                        
               
        # if there is no next page, we're done.
        if 'paging' in statuses.keys():
            statuses = json.loads(requestUntilSucceed(
                                        statuses['paging']['next']))
        else:
            has_next_page = False
            

    print "\nDone!\n%s Statuses Processed in %s" % \
                (num_processed, datetime.datetime.now() - scrape_starttime)
    
# store the information from the facebook page to a csv file    
def storeFacebookInformationCSV(page_name, access_token):
    with open('POSTagger\%s_facebook_statuses.csv' % page_name, 'wb') as file:
        w = csv.writer(file)
        w.writerow(["status_id", "status_message", "link_name", "status_type",
                    "status_link", "status_published", "num_reactions", 
                    "num_comments", "num_shares", "num_likes", "num_loves", 
                    "num_wows", "num_hahas", "num_sads", "num_angrys"])

        # has_next_page the variable that fetches data until everything is fetched
        has_next_page = True
        num_processed = 0   # keep a count on how many we've processed
        scrape_starttime = datetime.datetime.now()

        print "Scraping %s Facebook Page, at time: %s\n" % (page_name, scrape_starttime)

        # call the function getFacebookPageUrl
        statuses = getFacebookPageUrl(page_name, access_token, 100)

        #while has_next_page:
        while num_processed<100:            
            for status in statuses['data']:

                # Ensure it is a status with the expected metadata
                if 'reactions' in status:
                    w.writerow(processFacebookPageData(status,
                        access_token))

                # output progress occasionally to make sure code is not
                # stalling
                num_processed += 1
                if num_processed % 100 == 0:
                    print "%s Statuses Processed: %s" % \
                        (num_processed, datetime.datetime.now())

            # if there is no next page, we're done.
            if 'paging' in statuses.keys():
                statuses = json.loads(requestUntilSucceed(
                                        statuses['paging']['next']))
            else:
                has_next_page = False

        print "\nDone!\n%s Statuses Processed in %s" % \
                (num_processed, datetime.datetime.now() - scrape_starttime)

# insert data into the stanford tool to find which parts of a speech a status has
def analyzeTextStatus():

    cnx = connectDb()
    cnx2 = connectDb()
    cursor1 = cnx.cursor()
    query = ("SELECT status_id, status_message FROM statuses_pos ")

    cursor1.execute(query)

    for (status_id, status_message) in cursor1:
        with open('POSTagger\processed_output.txt', 'w') as fout:
            fout.write(status_message)
			
        subprocess.call('stanford-postagger models\wsj-0-18-left3words-distsim.tagger processed_output_twitter.txt > processed_output_POS_twitter.txt', cwd='POSTagger', shell=True)
		#subprocess.call('stanford-postagger models\wsj-0-18-left3words-distsim.tagger processed_output.txt > processed_output_POS.txt', shell=True)
        cc_counter, cd_counter, dt_counter, ex_counter, fw_counter, in_counter, jj_counter, jjr_counter, jjs_counter, ls_counter, md_counter, nn_counter, nns_counter, nnp_counter, nnps_counter, pdt_counter, pos_counter, prp_counter, prp6_counter, rb_counter, rbr_counter, rbs_counter, rp_counter, sym_counter, to_counter, uh_counter, vb_counter, vbd_counter,  vbg_counter, vbn_counter, vbp_counter, vbz_counter, wdt_counter, wp_counter, wp6_counter, wrb_counter = firstProcess()		
        
        if debug_variable:		
			print cc_counter, cd_counter, dt_counter, ex_counter, fw_counter, in_counter, jj_counter, jjr_counter, jjs_counter, ls_counter, md_counter, nn_counter, nns_counter, nnp_counter, nnps_counter, pdt_counter, pos_counter, prp_counter, prp6_counter, rb_counter, rbr_counter, rbs_counter, rp_counter, sym_counter, to_counter, uh_counter, vb_counter, vbd_counter,  vbg_counter, vbn_counter, vbp_counter, vbz_counter, wdt_counter, wp_counter, wp6_counter, wrb_counter
				
		# necessary changes because nn_counter includes nnp_counter and etc
        jj_counter = jj_counter - jjr_counter - jjs_counter
        nnp_counter = nnp_counter - nnps_counter
        nn_counter = nn_counter - nns_counter - nnp_counter - nnps_counter
        prp_counter = prp_counter - prp6_counter
        rb_counter = rb_counter - rbr_counter - rbs_counter
        wp_counter = wp_counter - wp6_counter
        		
        try:
            with cnx2.cursor() as cursor2:
                # create a new record
                # insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE statuses_pos
                           SET  num_cc=%s, num_cd=%s, num_dt=%s, num_ex=%s, num_fw=%s, num_in=%s, num_jj=%s, num_jjr=%s, num_jjs=%s, num_ls=%s, num_md=%s, num_nn=%s, num_nns=%s, num_nnp=%s, num_nnps=%s, num_pdt=%s, num_pos=%s, num_prp=%s, num_prp6=%s, num_rb=%s, num_rbr=%s, num_rbs=%s, num_rp=%s, num_sym=%s, num_to=%s, num_uh=%s, num_vb=%s, num_vbd=%s, num_vbg=%s, num_vbn=%s, num_vbp=%s, num_vbz=%s, num_wdt=%s, num_wp=%s, num_wp6=%s, num_wrb=%s
                           WHERE status_id=%s
                        """, (cc_counter, cd_counter, dt_counter, ex_counter, fw_counter, in_counter, jj_counter, jjr_counter, jjs_counter, ls_counter, md_counter, nn_counter, nns_counter, nnp_counter, nnps_counter, pdt_counter, pos_counter, prp_counter, prp6_counter, rb_counter, rbr_counter, rbs_counter, rp_counter, sym_counter, to_counter, uh_counter, vb_counter, vbd_counter,  vbg_counter, vbn_counter, vbp_counter, vbz_counter, wdt_counter, wp_counter, wp6_counter, wrb_counter, status_id) )
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)  
        
        cursor2.execute
		
    cursor1.close()
    cursor2.close()
    cnx.close()
    cnx2.close()	
	
# find the n most common words that are used in the statuses
def mostCommonWords():

    cnx = connectDb()
    cursor1 = cnx.cursor()

    cnx2 = connectDb()
    cursor2 = cnx.cursor()
    
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor1.execute(query)
    
    for (status_id, status_message) in cursor1:
        with open('POSTagger\processed_output_all.txt', 'a') as fout:
            fout.write(status_message)  
    
    with open ("POSTagger\processed_output_all.txt", "r") as myfile:
        data=myfile.read().replace('\n', ' ')

    data = data.split(' ')
    fdist1 = nltk.FreqDist(data)

    limit_of_recognized_words = 25
    most_common_words = fdist1.most_common(limit_of_recognized_words)

    with open('POSTagger\most_common_words.txt', 'a') as fout:
        for i in range(0,limit_of_recognized_words) :           
            fout.write(str(most_common_words[i][0])+"\n")   

            query2 = ("INSERT INTO words_frequency_pos " "(words,frequency)" "VALUES (%s, %s)") 
            insert_data = most_common_words[i][0], most_common_words[i][1]
            cursor2 = cnx2.cursor()
            try:
                with cnx2.cursor() as cursor2:
                    # Create a new record
                    #insert the data we pulled into db
                    cursor2.execute(query2, insert_data)
                    print("insert executions in words_frequency_pos is done")
                #commit the data to the db
                cnx2.commit()
            except ValueError as error:
                print(error)  

    cursor1.close()
    cnx.close()

    cursor2.close()
    cnx2.close()

# find how many times each of these frequent words appears
def frequencyWordsPOS():
    #subprocess.call('POSTagger\stanford-postagger models\wsj-0-18-left3words-distsim.tagger most_common_words.txt > most_common_words_POS.txt', shell=True)
    subprocess.call('stanford-postagger models\wsj-0-18-left3words-distsim.tagger processed_output_twitter.txt > processed_output_POS_twitter.txt', cwd='POSTagger', shell=True)

    # delete the fist 2 lines from the file most_common_words_POS
    with open('POSTagger\most_common_words_POS.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('POSTagger\most_common_words_POS.txt', 'w') as fout:
        fout.writelines(data[2:])

    # open write to the txt in a format that has one string in each line, better for the handling    
    text_into_a_variable = open('most_common_words_POS.txt').read()
    text_into_a_variable_each_string_a_line = text_into_a_variable.replace(' ','\n')
    with open('POSTagger\most_common_words_POS.txt', 'w') as fout:
        fout.write(text_into_a_variable_each_string_a_line)

    cnx = connectDb()
    cursor1 = cnx.cursor()
    query = ("SELECT words,frequency FROM words_frequency_pos ")
    cursor1.execute(query)
    cnx2 = connectDb()
    cursor2 = cnx.cursor()

    list_has_the_fetched_data_from_db = []
    for (words) in cursor1:
        list_has_the_fetched_data_from_db.append(words[0]) 

    with open('POSTagger\most_common_words_POS.txt') as f:
        for line in f:
            if debug_variable:
				print line
            for i in range(0, len(list_has_the_fetched_data_from_db)):                
                if list_has_the_fetched_data_from_db[i] in line:
                    if debug_variable:
						print line
						print list_has_the_fetched_data_from_db[i]
                    start =  line.find('_')
                    pos_var = line[start+1:]
                    if debug_variable:
						print pos_var

                    try:
                        with cnx2.cursor() as cursor2:
                            # Create a new record
                            # insert the data we pulled into db
                            cursor2.execute ("""
                                       UPDATE words_frequency_pos
                                       SET  pos=%s
                                       WHERE words=%s
                                    """, (pos_var, list_has_the_fetched_data_from_db[i]))
                            print("update executions is done")
                        # commit the data to the db
                        cnx2.commit()
                    except ValueError as error:
                        print(error)  

# call the dictionary that creates the machine score 
def machineScore():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)

    for (status_id, status_message) in cursor:
        my_list.append(status_message)
        if debug_variable:
			print status_message
        words = status_message.split()
        if debug_variable:
			print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            if debug_variable:
				print words[counter]
				print "\n"    
            with open('lexicons/AFINN.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
						print row[0]                    
                    if row[0] == words[counter]:
                        machine_result = machine_result + int(row[1])		
		if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                # insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE statuses_pos
                           SET  computer_score=%s
                           WHERE status_id=%s
                        """, (machine_result, status_id))        
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()

# call the dictionary from imdb that creates the machine score 
def machineScoreImdb():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)

    for (status_id, status_message) in cursor:
        my_list.append(status_message)
        if debug_variable:
			print status_message
        words = status_message.split()
        if debug_variable:
			print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            if debug_variable:
				print words[counter]
				print "\n"    
            with open('lexicons/final_version_of_imdb_lexicon.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
						print row[0]                    
                    if row[0] == words[counter]:
                        #print('---HERE----')
                        #print (row[3])																								  
                        machine_result = machine_result + float(row[3])		 
	    if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                # insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE statuses_pos
                           SET  computer_score_imdb=%s
                           WHERE status_id=%s
                        """, (machine_result, status_id))        
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()
	
# read the csv file with human score and update db
def humanScore():
	
    cnx = connectDb()
    cursor = cnx.cursor()    
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)

    for (status_id, status_message) in cursor:
    	if debug_variable:
			print status_message
    	
    	with open('lexicons/statuses_pos.csv', 'rb') as f:
		    reader = csv.reader(f)
		    for row in reader:
		        if debug_variable:
					print row[0]			        
		        if row[0] == status_id:
			    	if debug_variable:
						print row[0]
						print row[2]
		        	try:
				        with cnx2.cursor() as cursor2:
				            # Create a new record
				            # insert the data we pulled into db
				            cursor2.execute ("""
				                        UPDATE statuses_pos
				                        SET  human_score=%s
				                        WHERE status_id=%s
				                    """, (row[2], status_id))        
				            print("update executions is done")
				        # commit the data to the db
				        cnx2.commit()
		        	except ValueError as error:
				        print(error)  
        if debug_variable:								
			print "----------------------------"        
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()	

# read the datasheet and add Probabilities column				
def process_imdb_lexicon():

    # read data
    X = pd.read_csv('lexicons/imdb-words.csv')	
    pd.set_option('display.float_format', lambda x: '%.16f' % x)
    X['RelativeFrequency'] = X['Count']/X['Total']
    
    X['Probabilities'] = 0				    
    X = X.fillna(0)
    X = X.fillna('missing')    
    relFreqList = []
    sumRelFreq = 0
    j = 0
    for i in range(0,len(X)):
        sumRelFreq = 0									
        if (i+1)%10==0 :
            temp_X = X.loc[i-9:i]             
            sumRelFreq = temp_X['RelativeFrequency'].sum()
            relFreqList.append(sumRelFreq)
    for i in range(0, len(X)):
        X.loc[i,'Probabilities'] = X.loc[i,'RelativeFrequency']/relFreqList[j]
        if (i+1)%10==0 and i<30:
		   j = j + 1									
           												
    print(X.head(40))
    print(relFreqList)				
    #X.to_string(float_format='{:f}'.format)		
    X.to_csv('new_imdb.csv')

# add Expected Rate to the datasheet 				
def processImdbLexiconFindExpRate():
	
    #read data
    X = pd.read_csv('lexicons/imdb_RelFreq_Pr.csv')
    #print(X)	
    productPr = []			
    expRate = []				
    weights_list = [1,2,3,4,5,6,7,8,9,10]   
    j = 0
    X['ExpectedRate'] = X['Probabilities']				
    for i in range(0, len(X)):
        locVarSum = X['Probabilities'][i]*weights_list[j]
        j = j + 1
        productPr.append(locVarSum)								
        localSum = sum(productPr)	
        if (i+1)%10==0:
            #print(X['Word'][i])
            #print(localSum)
            expRate.append(localSum)												
            productPr = []
            j = 0									
            locVarSum = 0												
    #print(expRate)
    j = 0
    print('End of first for')				
    for i in range(0, len(X)):
        X['ExpectedRate'][i] = expRate[j]  					
        if (i+1)%10 == 0:
            j = j + 1									
    X.to_csv('new_imdb_with_ExpectedRate')

# modify the datasheet to have one row for (Word,Tag) without Probabilities, with Expected Rate
def modifyDataheet():
    X = pd.read_csv('lexicons/new_imdb_with_ExpectedRate.csv')
    #print (list(X.columns.values)	)
    X = X.drop(['Unnamed: 0','Unnamed: 0.1', 'RelativeFrequency', 'Probabilities', 'Category', 'Count', 'Total'], axis=1)				
    #print (X)
    for i in range(0, len(X)):
        if ((i+1)/10 == 0):
            myNewX = X.iloc[0::10,]
            #print ('in')	
    myNewX.index = range(0,len(myNewX))		
    myNewX.to_csv('lexicons/final_version_of_imdb_lexicon.csv')
  
# draw the dependence of the Relative Frequency
def depndecneRelFreq():
    X = pd.read_csv('lexicons/imdb_RelFreq_Pr.csv')
    for i in range(0, len(X)):				
        if (X['Word'][i] == 'bad' and X['Tag'][i] == 'a'):
            print (X['RelativeFrequency'][i])
        if (X['Word'][i] == 'horrible' and X['Tag'][i] == 'a'):
            print (X['Probabilities'][i])												
    #print (X.head(2))				

				
# multicorpus functions 
#####################
#####################
#####################
#####################				
				
# read the datasheet and add Probabilities column				
def process_multicorpus_lexicon():

    # read data
    #X = pd.read_csv('lexicons/amazon-tripadvisor.csv')	
    #X = pd.read_csv('lexicons/goodreads.csv')
    X = pd.read_csv('lexicons/opentable_only.csv')
					
    pd.set_option('display.float_format', lambda x: '%.16f' % x)
    X['RelativeFrequency'] = X['Count']/X['Total']
    
    X['Probabilities'] = 0				    
    X = X.fillna(0)
    X = X.fillna('missing')    
    relFreqList = []
    sumRelFreq = 0
    j = 0
    for i in range(0,len(X)):
        sumRelFreq = 0									
        if (i+1)%5==0 :
            temp_X = X.loc[i-4:i]             
            sumRelFreq = temp_X['RelativeFrequency'].sum()
            relFreqList.append(sumRelFreq)
    for i in range(0, len(X)):
        X.loc[i,'Probabilities'] = X.loc[i,'RelativeFrequency']/relFreqList[j]
        if (i+1)%5==0 :
		   j = j + 1									
           												
    #print(X.head(40))
    #print(relFreqList)				
    #X.to_string(float_format='{:f}'.format)		
    #X.to_csv('lexicons/new_amazon-tripadvisor.csv')
    #X.to_csv('lexicons/new_goodreads.csv')				
    X.to_csv('lexicons/new_opentable.csv')				

# add Expected Rate to the datasheet 				
def processMulticorpusLexiconFindExpRate():
	
    #read data
    #X = pd.read_csv('lexicons/new_amazon-tripadvisor.csv')
    #X = pd.read_csv('lexicons/new_goodreads.csv')
    X = pd.read_csv('lexicons/new_opentable.csv')
								
    #print(X)	
    productPr = []			
    expRate = []				
    weights_list = [1,2,3,4,5]   
    j = 0
    X['ExpectedRate'] = X['Probabilities']				
    for i in range(0, len(X)):
        locVarSum = X['Probabilities'][i]*weights_list[j]
        j = j + 1
        productPr.append(locVarSum)								
        localSum = sum(productPr)	
        if (i+1)%5==0:
            #print(X['Word'][i])
            #print(localSum)
            expRate.append(localSum)												
            productPr = []
            j = 0									
            locVarSum = 0												
    #print(expRate)
    j = 0
    print('End of first for')				
    for i in range(0, len(X)):
        X['ExpectedRate'][i] = expRate[j]  					
        if (i+1)%5 == 0:
            j = j + 1
        if (i+1)%1000 == 0:	
            print 'woop'												
    #X.to_csv('lexicons/new_amazon-tripadvisor_ExpectedRate.csv')
    #X.to_csv('lexicons/new_goodreads_ExpectedRate.csv')
    X.to_csv('lexicons/new_opentable_ExpectedRate.csv')

				
# modify the datasheet to have one row for (Word,Tag) without Probabilities, with Expected Rate
def modifyMultiCorpusDataheet():
    #X = pd.read_csv('lexicons/new_amazon-tripadvisor_ExpectedRate.csv')
    #X = pd.read_csv('lexicons/new_goodreads_ExpectedRate.csv')				
    X = pd.read_csv('lexicons/new_opentable_ExpectedRate.csv')				
				
    print (list(X.columns.values)	)
    				
    X = X.drop(['Unnamed: 0', 'Rating', 'Corpus', 'RelativeFrequency', 'Probabilities', 'Category', 'Count', 'Total'], axis=1)				
    print (X.head(2))
    for i in range(0, len(X)):
        if ((i+1)/5 == 0):
            myNewX = X.iloc[0::5,]
            #print ('in')	
    myNewX.index = range(0,len(myNewX))		
    #myNewX.to_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon.csv')
    #myNewX.to_csv('lexicons/final_version_of_goodreads_lexicon.csv')				
    myNewX.to_csv('lexicons/final_version_of_opentable_lexicon.csv')				

				
				
# remove a and r from the word in multicorpus
def removeType():
    #X = pd.read_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon.csv')
    #X = pd.read_csv('lexicons/final_version_of_goodreads_lexicon.csv')				
    X = pd.read_csv('lexicons/final_version_of_opentable_lexicon.csv')
				
    X['Word'] = X['Word'].str.split('/').str.get(0)
    print X['Word']							

    #X.to_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon_without_type.csv')								
    #X.to_csv('lexicons/final_version_of_goodreads_lexicon_without_type.csv')
    X.to_csv('lexicons/final_version_of_opentable_lexicon_without_type.csv')
				
				

# remove rows which contain the same word				
def removeDuplicatesFromMulticorpus():
	
    #X = pd.read_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon_without_type.csv')
    X = pd.read_csv('lexicons/final_version_of_goodreads_lexicon_without_type.csv')

    X = X.sort_values(['Word', 'ExpectedRate'], ascending=[True, False])
				
    X = X.drop_duplicates(['Word'])
    #X.to_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon_without_type_without_duplicates.csv')
    X.to_csv('lexicons/final_version_of_goodreads_lexicon_without_type_without_duplicates.csv')
				
# remove duplicates and get the average				
def removeDuplicatesFromMulticorpusGetAverage():

    #X = pd.read_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon_without_type.csv')
    #X = pd.read_csv('lexicons/final_version_of_goodreads_lexicon_without_type.csv')
    X = pd.read_csv('lexicons/final_version_of_opentable_lexicon_without_type.csv')
				
    X['avg'] = X['ExpectedRate']				
    #X['avg'] = 0.0				
    for i in range(1,len(X)):
    #for i in range(90,95):
					
        if i%1000 == 0 :
            print 'woop'									
        
        if X['Word'][i] == X['Word'][i-1] : 
            #print 'aa'           
            X.loc[i,'avg'] = float((X['ExpectedRate'][i] + X['ExpectedRate'][i-1])/2)												
            X.loc[i-1,'avg'] = float((X['ExpectedRate'][i] + X['ExpectedRate'][i-1])/2)
            #print X['Word'][i]
            #print X['ExpectedRate'][i]
            #print X['ExpectedRate'][i-1]
            #print X['avg'][i]												
												
    X = X.drop_duplicates(['Word'])									
				
    #X.to_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon_without_type_without_duplicates_avg.csv')
    #X.to_csv('lexicons/final_version_of_goodreads_lexicon_without_type_without_duplicates_avg.csv')
    X.to_csv('lexicons/final_version_of_opentable_lexicon_without_type_without_duplicates_avg.csv')
				
				
# call the dictionary from amazon-tripadvisor that creates the machine score 
def machineScoreMultiCorpus():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)
    
    for (status_id, status_message) in cursor:
        my_list.append(status_message)
        if debug_variable:
            print status_message
        words = status_message.split()
        if debug_variable:
            print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
        #for counter in range (0,5):									
            #raw_input()									
            if debug_variable:
                print words[counter]
                #rint "\n"    
            #with open('multicorpus/final_version_of_amazon-tripadvisor_lexicon_without_type.csv', 'rb') as f:
            #with open('multicorpus/final_version_of_goodreads_lexicon_without_type.csv', 'rb') as f:													
            with open('multicorpus/final_version_of_opentable_lexicon_without_type.csv', 'rb') as f:													
													
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
                        print row[0]                    
                    if row[0] == words[counter]:
                        #print('---HERE----')
                        #print 'row[0] : ' + row[0]
                        #print (row[1])																								  
                        machine_result = machine_result + float(row[1])		 
	    if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                # insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE statuses_pos
                           SET  computer_score_opentable=%s
                           WHERE status_id=%s
                        """, (machine_result, status_id))        
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()
				
#####################
#####################				
#####################
#####################				
				

#####################
#####################				
#####################
#####################				

# wnscores - inquirer

def dropZeros():

    X = pd.read_csv('lexicons/wnscores_inquirer.csv')
    print X.head(4)			
    X = X.loc[(X.Score!=0)]
    print X.head(4)	
    print len(X)													
    X.to_csv('lexicons/wnscores_inquirer_without_zeros.csv')
				

def getAvgWnInquirer():
    
    X = pd.read_csv('lexicons/wnscores_inquirer_without_zeros.csv')
    doubles = []	
    loc_sum = 0
    loc_counter = 1
    bool_var = False				
    X['avg'] = X['Score']				
    for i in range(1,len(X)):					
        if i%1000 == 0 :
            print 'woop'										
        if bool_var == False:												
            loc_sum = X['Score'][i-1]												
            loc_counter = 1
												
        if X['Word'][i] == X['Word'][i-1] :            									
            doubles.append(X['Word'][i])									
            loc_sum = loc_sum + X['Score'][i]
            loc_counter = loc_counter + 1
            bool_var = True
            for j in range(i-loc_counter+1, i ):												
                X['avg'][j] = loc_sum/loc_counter
            X['avg'][i] = loc_sum/loc_counter																								
        else :     									
            bool_var = False
												
    #X = X.drop_duplicates(['Word'])									
    print len(doubles)				
    X.to_csv('lexicons/wnscores_inquirer_without_zeros_avg.csv')
				
def dropDuplicates():

    X = pd.read_csv('lexicons/wnscores_inquirer_without_zeros_avg.csv')
    print X[X['Word'].duplicated(keep=False)]												
    X['ScoreAbs'] = X.Score.abs()																							
    X = X.sort_values(['Word', 'ScoreAbs'], ascending=[True, False])				
    X = X.drop_duplicates(['Word'])    				    
    X = X.drop(['Unnamed: 0','Unnamed: 0.1','ScoreAbs'], axis=1)				
    X.to_csv('lexicons/wnscores_inquirer_without_zeros_avg_without_duplicates.csv')
	

def testLexicon():				
				
    X = pd.read_csv('lexicons/wnscores_inquirer_without_zeros_avg_without_duplicates.csv')
    X = X.drop('Unnamed: 0', axis=1)				
    #print X.head(3)
    x_unique =  X['Score'].unique()
    print 'wnscores_inquirer'				
    print 'number of values: '								
    print len(x_unique)
    #print 'frequency'				
    #print X['Score'].value_counts()					
    print 'Mean : '				
    print X['Score'].mean()	
    print 'Variance : '				
    print X['Score'].var()
				
    print '-------------------------------------------------------'
    print 'amazon-tripadvisor'				
    X1 = pd.read_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon_without_type_without_duplicates_avg.csv')
    x_unique =  X1['avg'].unique()
    print 'number of values: '				
    print len(x_unique)
    #print 'frequency'				
    #print X1['avg'].value_counts()					
    print 'Mean : '				
    print X1['avg'].mean()	
    print 'Variance : '				
    print X['avg'].var()
				
    print '-------------------------------------------------------'
    print 'goodreads'			
    X1 = pd.read_csv('lexicons/final_version_of_goodreads_lexicon_without_type_without_duplicates_avg.csv')
    x_unique =  X1['avg'].unique()
    print 'number of values: '				
    print len(x_unique)
    #print 'frequency'				
    #print X1['avg'].value_counts()					
    print 'Mean : '				
    print X1['avg'].mean()	
    print 'Variance : '				
    print X['avg'].var()
				
    print '-------------------------------------------------------'
    print 'imdb'			
    X1 = pd.read_csv('lexicons/final_version_of_imdb_lexicon.csv')
    x_unique =  X1['ExpectedRate'].unique()
    print 'number of values: '				
    print len(x_unique)
    #print 'frequency'				
    #print X1['ExpectedRate'].value_counts()					
    print 'Mean : '				
    print X1['ExpectedRate'].mean()	
    print 'Variance : '				
    print X1['ExpectedRate'].var()				

    print '-------------------------------------------------------'
    print 'opentable'			
    X1 = pd.read_csv('lexicons/final_version_of_opentable_lexicon_without_type.csv')
    x_unique =  X1['ExpectedRate'].unique()
    print 'number of values: '				
    print len(x_unique)
    #print 'frequency'				
    #print X1['ExpectedRate'].value_counts()					
    print 'Mean : '				
    print X1['ExpectedRate'].mean()	
    print 'Variance : '				
    print X1['ExpectedRate'].var()				
				
    print '-------------------------------------------------------'
    print 'sentiwordnet'			
    X1 = pd.read_csv('lexicons/senti_word_net_lexicon_v2.csv')
    X1 = X1.loc[(X1!=0).any(1)]				
    x_unique =  X1['score'].unique()
    print 'number of values: '				
    print len(x_unique)
    #print 'frequency'				
    #print X1['Score'].value_counts()					
    print 'Mean : '				
    print X1['score'].mean()	
    print 'Variance : '				
    print X1['score'].var()				
				
    print '-------------------------------------------------------'
    print 'subjective'			
    X1 = pd.read_csv('lexicons/SubjectiveLexicon2.csv')
    X1 = X1.loc[(X1!=0).any(1)]				
    x_unique =  X1['score'].unique()
    print 'number of values: '				
    print len(x_unique)
    #print 'frequency'				
    #print X1['score'].value_counts()				
    print 'Mean : '				
    print X1['score'].mean()	
    print 'Variance : '				
    print X1['score'].var()					
    print '-------------------------------------------------------'
				
# call the dictionary from wnscores_inquirer that creates the machine score 
def machineScoreWnscoresInquirer():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)

    for (status_id, status_message) in cursor:
        my_list.append(status_message)
        if debug_variable:
			print status_message
        words = status_message.split()
        if debug_variable:
			print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            if debug_variable:
				print words[counter]
				print "\n"    
            with open('wnscores_inquirer/wnscores_inquirer_without_zeros_avg.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
						print row[0]                    
                    if row[0] == words[counter]:
                        #print('---HERE----')
                        #print (row[3])																								  
                        machine_result = machine_result + float(row[4])		 
	    if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                # insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE statuses_pos
                           SET  computer_score_wnscores =%s
                           WHERE status_id=%s
                        """, (machine_result, status_id))        
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()
				
				
# count the rows have been scored
def countWnscoresInquirer():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)
    loc_bool_var = False
    loc_counter = 0			
    for (status_id, status_message) in cursor:
        my_list.append(status_message)
        words = status_message.split()
        machine_result = 0
        if loc_bool_var==True :
            loc_counter = loc_counter + 1
        loc_bool_var = False												
        for counter in range (0,len(words)):
            if debug_variable:
				print words[counter]
				print "\n"    
            with open('lexicons/senti_word_net_lexicon_v2.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:                
                    if row[0] == words[counter]:																							  
                        machine_result = machine_result + float(row[1])	
                        loc_bool_var = True  

    print loc_counter          
    cursor.close()
    cnx.close()
    cnx2.close()				
#####################
#####################				
#####################
#####################				


# machineScorePosNeg
def machineScorePosNeg():
	
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)

    for (status_id, status_message) in cursor:
        my_list.append(status_message)
        if debug_variable:
			print status_message
        words = status_message.split()
        if debug_variable:
			print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            if debug_variable:
				print words[counter]
				print "\n"    
            with open('pos-neg-words/positive-words.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
						print row[0]                    
                    if row[0] == words[counter]:
                        machine_result = machine_result + 8
            with open('pos-neg-words/negative-words.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
						print row[0]                    
                    if row[0] == words[counter]:
                        machine_result = machine_result - 8
																								
		if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                # insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE statuses_pos
                           SET  computer_score_pos_neg=%s
                           WHERE status_id=%s
                        """, (machine_result, status_id))        
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()

# process the lexicon SentiWordNet				
def processSentiWordNet():

    SentiWordNetLexicon = []	
    SentiWordNetLexiconScore = []
    only_one_word = []
    score_for_only_one_word = []
    filename = 'lexicons/SentiWordNet.txt'
    with open(filename) as f:
        data = f.readlines()

    for i in range(28, len(data)):
        string_data = ''.join(data[i])
        sublist = re.split(r'\t+', string_data)																
        SentiWordNetLexicon.append(sublist[4])								
        score = float(sublist[2]) - float(sublist[3])							
        SentiWordNetLexiconScore.append(score)
								
    for i in range(0, len(SentiWordNetLexicon)):
        positions = [pos for pos, char in enumerate(SentiWordNetLexicon[i]) if char == '#']
        one_line_of_word = SentiWordNetLexicon[i]																					 
        for j in range(0,len(positions)):
            if j>0:	
                only_one_word.append(one_line_of_word[positions[j-1]+3:positions[j]])
                score_for_only_one_word.append(SentiWordNetLexiconScore[i])														
            else:
                only_one_word.append(one_line_of_word[:positions[j]])
                score_for_only_one_word.append(SentiWordNetLexiconScore[i]) 															  
    
    resultFile = open("lexicons/output.csv",'wb')
    wr = csv.writer(resultFile, dialect='excel')
    for i in range (0, len(only_one_word)):	
        wr.writerow([only_one_word[i],score_for_only_one_word[i]])	
								
# process the Subjective
def machineScoreSubjective():
	
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)

    for (status_id, status_message) in cursor:
        my_list.append(status_message)
        if debug_variable:
			print status_message
        words = status_message.split()
        if debug_variable:
			print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            if debug_variable:
				print words[counter]
				print "\n"    
            with open('subjectivity_clues/SubjectiveLexicon2.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
						print row[3]                    
                    if row[3] == words[counter]:
                        #print('---HERE----')
                        #print (row[3])																								  
                        machine_result = machine_result + int(row[7])		 
	    if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                # insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE statuses_pos
                           SET  computer_score_subjective=%s
                           WHERE status_id=%s
                        """, (machine_result, status_id))        
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()	


# call the dictionary from imdb that creates the machine score 
def machineScoreSentiWordNet():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT status_id, status_message FROM statuses_pos ")
    cursor.execute(query)

    for (status_id, status_message) in cursor:
        my_list.append(status_message)
        if debug_variable:
			print status_message
        words = status_message.split()
        if debug_variable:
			print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            if debug_variable:
				print words[counter]
				print "\n"    
            with open('lexicons/senti_word_net_lexicon_v2.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
						print row[0]                    
                    if row[0] == words[counter]:
                        #print('---HERE----')
                        #print (row[3])																								  
                        machine_result = machine_result + float(row[1])		 
	    if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                # insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE statuses_pos
                           SET  computer_score_senti_word_net=%s
                           WHERE status_id=%s
                        """, (machine_result, status_id))        
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()

								
								
if __name__ == '__main__':

    #storeFacebookInformationCSV(page_name, access_token)
    #storeFacebookInformationDataBase(page_name, access_token)    
    #analyzeTextStatus()    
    #mostCommonWords()
    #frequencyWordsPOS()
    #machineScore()
    #process_imdb_lexicon()
    #processImdbLexiconFindExpRate()				
    #humanScore()
    #modifyDataheet()
    #machineScoreImdb()
    #depndecneRelFreq()	
    #machineScorePosNeg()
    #processSentiWordNet()
    #machineScoreSentiWordNet()
    #machineScoreSubjective()
				
    #process_multicorpus_lexicon()	
    #processMulticorpusLexiconFindExpRate()				
    #modifyMultiCorpusDataheet()				
    #removeType()	
    #removeDuplicatesFromMulticorpus()
    #removeDuplicatesFromMulticorpusGetAverage()				
    #machineScoreMultiCorpus()
				
    #getAvgWnInquirer()				
    #dropZeros()
    #dropDuplicates()			
    #machineScoreWnscoresInquirer()
    testLexicon()				
    #countWnscoresInquirer()     				
    print 'bye'

'''
    if debug_variable:
        print "thesis"
    else :
        print 'end of code'
'''								
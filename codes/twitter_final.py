# twitter-page-timeline
# displays the a user's current timeline.
import mysql.connector
import pymysql.cursors
import urllib2
import json
import datetime
import csv
import time
import mysql.connector
import pymysql.cursors
import unicodedata
import pandas as pd
import subprocess
import re
import nltk
from twitter import *

# boolean debug variable
debug_variable = False 

# load API credentials, using external file 
# using external file in favor of abstractness 
config = {}
execfile("config.py", config)

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

    file=open("POSTagger\processed_output_POS_twitter.txt","r+")
    
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

# collect data from twitter and insert them into database
def fetchAndStore():
	
	# create twitter API object
	# using the external file
	twitter = Twitter(
			auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))
	
	# this is the page we're going to query.
	user = "asus"

	# query the user timeline.
	results = twitter.statuses.user_timeline(screen_name = user, count = 10)

	# loop through each status item
	# insert to database
	for status in results:
		if debug_variable:
			print "%d (%s) %s %d %d" % (status["id"], status["created_at"], status["text"].encode("ascii", "ignore"), status["retweet_count"], status["favorite_count"])
		
		# SQL statement for adding Twitter data
		insert_info = ("INSERT INTO tweet_data " "(tweet_id, created_at, tweet_text, retweet_count, favorite_count)" "VALUES (%s, %s, %s, %s,%s)") 
	    
		id_db = status["id"]
		created_at_db = status["created_at"]
		text_db = status["text"].encode("ascii", "ignore")
		retweet_count_db = status["retweet_count"]
		favorite_count_db = status["favorite_count"]
		page_data = id_db, created_at_db, text_db, retweet_count_db, favorite_count_db
		connection = connectDb()
		cursor = connection.cursor()
	    
		try:
			with connection.cursor() as cursor:
	            # Create a new record
	            # insert the data we pulled into db
				cursor.execute(insert_info, page_data)
				if debug_variable:    
					print("insert into tweet_data is done")
	        # commit the data to the db
			connection.commit()
		except ValueError as error:
			print(error)   
	
	cursor.close()
	connection.commit()		
                    
# fetch data from database
# use POSTagger tool					
def fetchAndUpdate():

    cnx = connectDb()
    cnx2 = connectDb()
    cursor1 = cnx.cursor()
    cursor2 = cnx2.cursor()
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
    cursor1.execute(query)
    
    for (tweet_id, tweet_text) in cursor1:
        with open('POSTagger\processed_output_twitter.txt', 'w') as fout:
            fout.write(tweet_text)
        
        subprocess.call('stanford-postagger models\wsj-0-18-left3words-distsim.tagger processed_output_twitter.txt > processed_output_POS_twitter.txt', cwd='POSTagger', shell=True)
        cc_counter, cd_counter, dt_counter, ex_counter, fw_counter, in_counter, jj_counter, jjr_counter, jjs_counter, ls_counter, md_counter, nn_counter, nns_counter, nnp_counter, nnps_counter, pdt_counter, pos_counter, prp_counter, prp6_counter, rb_counter, rbr_counter, rbs_counter, rp_counter, sym_counter, to_counter, uh_counter, vb_counter, vbd_counter,  vbg_counter, vbn_counter, vbp_counter, vbz_counter, wdt_counter, wp_counter, wp6_counter, wrb_counter = firstProcess()		
        
        if debug_variable:
			print cc_counter, cd_counter, dt_counter, ex_counter, fw_counter, in_counter, jj_counter, jjr_counter, jjs_counter, ls_counter, md_counter, nn_counter, nns_counter, nnp_counter, nnps_counter, pdt_counter, pos_counter, prp_counter, prp6_counter, rb_counter, rbr_counter, rbs_counter, rp_counter, sym_counter, to_counter, uh_counter, vb_counter, vbd_counter,  vbg_counter, vbn_counter, vbp_counter, vbz_counter, wdt_counter, wp_counter, wp6_counter, wrb_counter 
        
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                #insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE tweet_data
                           SET  num_cc=%s, num_cd=%s, num_dt=%s, num_ex=%s, num_fw=%s, num_in=%s, num_jj=%s, num_jjr=%s, num_jjs=%s, num_ls=%s, num_md=%s, num_nn=%s, num_nns=%s, num_nnp=%s, num_nnps=%s, num_pdt=%s, num_pos=%s, num_prp=%s, num_prp6=%s, num_rb=%s, num_rbr=%s, num_rbs=%s, num_rp=%s, num_sym=%s, num_to=%s, num_uh=%s, num_vb=%s, num_vbd=%s, num_vbg=%s, num_vbn=%s, num_vbp=%s, num_vbz=%s, num_wdt=%s, num_wp=%s, num_wp6=%s, num_wrb=%s
                           WHERE tweet_id=%s
                        """, (cc_counter, cd_counter, dt_counter, ex_counter, fw_counter, in_counter, jj_counter, jjr_counter, jjs_counter, ls_counter, md_counter, nn_counter, nns_counter, nnp_counter, nnps_counter, pdt_counter, pos_counter, prp_counter, prp6_counter, rb_counter, rbr_counter, rbs_counter, rp_counter, sym_counter, to_counter, uh_counter, vb_counter, vbd_counter,  vbg_counter, vbn_counter, vbp_counter, vbz_counter, wdt_counter, wp_counter, wp6_counter, wrb_counter, tweet_id) )
                print("update executions is done")
            # commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)  
			
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
    
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
    cursor1.execute(query)
    
    for (tweet_id, tweet_text) in cursor1:
        with open('POSTagger\processed_output_all.txt', 'a') as fout:
            fout.write(tweet_text)  
    
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
    subprocess.call('stanford-postagger models\wsj-0-18-left3words-distsim.tagger processed_output_twitter.txt > processed_output_POS_twitter.txt', cwd='POSTagger', shell=True)

    #delete the fist 2 lines from the file most_common_words_POS
    with open('POSTagger\most_common_words_POS.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('POSTagger\most_common_words_POS.txt', 'w') as fout:
        fout.writelines(data[2:])

    #open write to the txt in a format that has one string in each line, better for the handling    
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
            #print line
            for i in range(0, len(list_has_the_fetched_data_from_db)):                
                if list_has_the_fetched_data_from_db[i] in line:
                    #print line
                    print list_has_the_fetched_data_from_db[i]
                    start =  line.find('_')
                    pos_var = line[start+1:]
                    print pos_var

                    try:
                        with cnx2.cursor() as cursor2:
                            # Create a new record
                            #insert the data we pulled into db
                            cursor2.execute ("""
                                       UPDATE words_frequency_pos
                                       SET  pos=%s
                                       WHERE words=%s
                                    """, (pos_var, list_has_the_fetched_data_from_db[i]))
                            print("update executions is done")
                        #commit the data to the db
                        cnx2.commit()
                    except ValueError as error:
                        print(error)  
    
# call the dictionary that creates the machine score 
def machineScore():
    
    cnx = connectDb()
    cursor = cnx.cursor()    
    cnx2 = connectDb()
    cursor2 = cnx.cursor()
    my_list = []
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
    cursor.execute(query)

    for (tweet_id, tweet_text) in cursor:

        my_list.append(tweet_text)
        print tweet_text
        words = tweet_text.split()
        #print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            #print words[counter]
            #print "\n"
    
            with open('lexicons/AFINN.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    #print row[0]                    
                    if row[0] == words[counter]:
                        machine_result = machine_result + int(row[1])
        print machine_result    

        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                #insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE tweet_data
                           SET  computer_score=%s
                           WHERE tweet_id=%s
                        """, (machine_result, tweet_id))        
                print("update executions is done")
            #commit the data to the db
            cnx2.commit()
        except Error as error:
            print(error)  
        
        cursor2.execute

    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()

# call the dictionary that creates the machine score 
def machineScoreImdb():
    
    cnx = connectDb()
    cursor = cnx.cursor()    
    cnx2 = connectDb()
    cursor2 = cnx.cursor()
    my_list = []
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
    cursor.execute(query)

    for (tweet_id, tweet_text) in cursor:
        my_list.append(tweet_text)
        print tweet_text
        words = tweet_text.split()
        #print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            #print words[counter]
            #print "\n"   
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
                #insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE tweet_data
                           SET  computer_score_imdb=%s
                           WHERE tweet_id=%s
                        """, (machine_result, tweet_id))        
                print("update executions is done")
            #commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()
	


# call the dictionary that creates the machine score from the dictionaries that have only positive or negative words
def machineScorePosNeg():
    
    cnx = connectDb()
    cursor = cnx.cursor()    
    cnx2 = connectDb()
    cursor2 = cnx.cursor()
    my_list = []
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
    cursor.execute(query)

    for (tweet_id, tweet_text) in cursor:
        my_list.append(tweet_text)
        if debug_variable:
            print tweet_text
        words = tweet_text.split()
        #print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            #print words[counter]
            #print "\n"   
            with open('lexicons/positive-words.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
                        print row[0]                    
                    if row[0] == words[counter]:
                        #print('---HERE----')
                        #print (row[3])																								  
                        machine_result = machine_result + 1
																								
        for counter in range (0,len(words)):
            #print words[counter]
            #print "\n"   
            with open('lexicons/negative-words.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
                        print row[0]                    
                    if row[0] == words[counter]:
                        #print('---HERE----')
                        #print (row[3])																								  
                        machine_result = machine_result - 1
																								
		if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                #insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE tweet_data
                           SET  computer_score_pos_neg=%s
                           WHERE tweet_id=%s
                        """, (machine_result, tweet_id))        
                print("update executions is done")
            #commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()
	
def machineScoreSentiWordLexicon():				

    cnx = connectDb()
    cursor = cnx.cursor()    
    cnx2 = connectDb()
    cursor2 = cnx.cursor()
    my_list = []
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
    cursor.execute(query)

    for (tweet_id, tweet_text) in cursor:
        my_list.append(tweet_text)
        #print tweet_text
        words = tweet_text.split()
        #print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            #print words[counter]
            #print "\n"   
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
                #insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE tweet_data
                           SET  computer_score_senti_word_net=%s
                           WHERE tweet_id=%s
                        """, (machine_result, tweet_id))        
                print("update executions is done")
            #commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()
				
	
def machineScoreSubjectivityLexicon():				

    cnx = connectDb()
    cursor = cnx.cursor()    
    cnx2 = connectDb()
    cursor2 = cnx.cursor()
    my_list = []
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
    cursor.execute(query)

    for (tweet_id, tweet_text) in cursor:
        my_list.append(tweet_text)
        #print tweet_text
        words = tweet_text.split()
        #print len(words)
        machine_result = 0
        for counter in range (0,len(words)):
            #print words[counter]
            #print "\n"   
            with open('lexicons/SubjectiveLexicon.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    if debug_variable:
                        print row[0]                    
                    if row[0] == words[counter]:
                        #print('---HERE----')
                        #print (row[3])																								  
                        machine_result = machine_result + int(row[7])		 
		if debug_variable:
			print machine_result    
        try:
            with cnx2.cursor() as cursor2:
                # Create a new record
                #insert the data we pulled into db
                cursor2.execute ("""
                           UPDATE tweet_data
                           SET  computer_score_subjectivity=%s
                           WHERE tweet_id=%s
                        """, (machine_result, tweet_id))        
                print("update executions is done")
            #commit the data to the db
            cnx2.commit()
        except ValueError as error:
            print(error)          
        cursor2.execute
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()

# call the dictionary from amazon-tripadvisor that creates the machine score 
def machineScoreMulticorpus():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
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
            with open('lexicons/final_version_of_opentable_lexicon_without_type.csv', 'rb') as f:
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
                           UPDATE tweet_data
                           SET  computer_score_opentable =%s
                           WHERE tweet_id=%s
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

		
				
# call the dictionary from wnscores_inquirer that creates the machine score 
def machineScoreWnscoresInquirer():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
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
            with open('lexicons/wnscores_inquirer_without_zeros_avg.csv', 'rb') as f:
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
                           UPDATE tweet_data
                           SET  computer_score_wnscores =%s
                           WHERE tweet_id=%s
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
				
#########################
#########################
#########################		
# count the rows have been scored
def countWnscoresInquirer():
    
    cnx = connectDb()
    cursor = cnx.cursor()
    cnx2 = connectDb()
    my_list = []
    query = ("SELECT tweet_id, tweet_text FROM tweet_data ")
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
            with open('lexicons/wnscores_inquirer.csv', 'rb') as f:
                reader = csv.reader(f)
                for row in reader:                
                    if row[0] == words[counter]:																							  
                        machine_result = machine_result + float(row[2])	
                        loc_bool_var = True  

    print loc_counter          
    cursor.close()
    cnx.close()
    cnx2.close()	

#########################
#########################
#########################

								
# read the csv file with human score and update db
def humanScore():
	
	
    cnx = connectDb()
    cursor = cnx.cursor()   
    cnx2 = connectDb()
    cursor2 = cnx.cursor()
	
    my_list = []
    query = ("SELECT tweet_text, created_at FROM tweet_data ")
    cursor.execute(query)

    for (tweet_text, created_at) in cursor:
		if debug_variable:
		    print tweet_text
    	   	
		with open('lexicons/tweet_data_final.csv', 'rb') as f:
		    reader = csv.reader(f)
		    for row in reader:
			    if debug_variable:
					print row[3]		        
			    if row[1] == created_at:
		        	if debug_variable:
					    print row[0]
					    print row[2]
		        	try:
				        with cnx2.cursor() as cursor2:
				            # Create a new record
				            # insert the data we pulled into db
				            cursor2.execute ("""
				                        UPDATE tweet_data
				                        SET  human_score=%s
				                        WHERE created_at=%s
				                    """, (row[3], created_at))        
				            print("update executions is done")
				        # commit the data to the db
				        cnx2.commit()
		        	except ValueError as error:
				        print(error)  
    	#print "----------------------------"       
    cursor.close()
    cursor2.close()
    cnx.close()
    cnx2.close()	
				
				
if __name__ == '__main__':
    #fetchAndStore()
    #fetchAndUpdate()
    #mostCommonWords()
    #frequencyWordsPOS()
    #machineScore()
    #machineScoreImdb()				
    #humanScore()
    #machineScorePosNeg()
    #machineScoreSentiWordLexicon()				
    #machineScoreSubjectivityLexicon()
    #machineScoreMulticorpus()
    #machineScoreWnscoresInquirer()
    #countWnscoresInquirer()				
				
    if debug_variable:
		print "thesis"
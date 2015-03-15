import pandas as pd
import numpy as np
import psycopg2
import urllib 

# class Loading_data(object):

# 	def __init__(self, )

#merge dataframes on "desc"(there are two csv files for each label) 
# def merge_df(dir='beach'):
# 	file1_name = dir + '_1.csv'
# 	file2_name = dir + '_2.csv'
#     df1 = pd.read_csv(file1_name)
#     df2 = pd.read_csv(file2_name)
# 	df1.columns = ['desc', 'img_url', 'image_href', 'img_text']	
# 	df2.columns = ['img_href', 'title', 'desc_href', 'desc']
# 	df = df1.merge(df2, on="desc")
#     return df
	#assuming I am using data from the pin_meta data files (only 1 file)
def process_dataframe(directory='beach'):
	filename = 'data/' + directory + 'data.csv'
	df = pd.read_csv(filename)
	"large_image.alt","large_image.src","large_image.href","large_image.text","title","tag_desc","ogi_pinner","board_name.href","board_name.text"
	df.columns = ['pin_desc', 'img_url', 'img_href', 'img_text', 'title', 'tag_desc', 'orig_pinner', 'board_name_href', 'board_name']
	
	df.apply(lambda x: x.replace('\'', ''))
	return df

def save_image(dataframe, directory='beach'):
	conn = psycopg2.connect(dbname='pinterest_wedding', user='postgres', host='/tmp')
	c = conn.cursor()

	c.execute('''
		CREATE TABLE pin_data 
		(
			pin_id				varchar(100),
			category           varchar(100),
			pin_desc			varchar(5000),
			img_url				varchar(5000),
			img_href 			varchar(5000),
			img_text 			varchar(5000),
			title 				varchar(5000),  			
			tag_desc 			varchar(5000),
			orig_pinner 		varchar(5000),
			board_name_href 	varchar(5000),
			board_name 			varchar(5000),
			PRIMARY KEY(pin_id)
		)''')

    # rows = c.fetchall()
	for i, row in enumerate(df.iterrows()):
		row = row[1]
		pin_id = directory + str(i)
		category = directory
		pin_desc = row['pin_desc']
		img_url = row['img_url']
		img_href = row['img_href']
		img_text = row['img_text']
		title = row['title']
		tag_desc = row['tag_desc']
		orig_pinner = row['orig_pinner']
		board_name_href = row['board_name_href']
		board_name = row['board_name']
		c.execute('''INSERT INTO pin_data VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''' % 
				(pin_id, category, pin_desc, img_url, img_href, img_text,
				title tag_desc, orig_pinner, board_name_href, board_name))
	conn.commit()
	conn.close()

	# Writing the images
	resource = urllib.urlopen(img_url)
	dir_url = 'img_dir/' + directory + '/' + pin_id + '.jpg'
	output = open(dir_url, "wb")
	output.write(resource.read())
	output.close()

if __name__=='__main__':
	df = process_dataframe('beach')
	save_image(df, directory='beach')

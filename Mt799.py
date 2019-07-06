from __future__ import division
import nltk
from nltk import PunktSentenceTokenizer, RegexpTokenizer, word_tokenize, bigrams
from itertools import repeat, chain
import difflib
import re
import os
import csv
import json
import tempfile
import requests
import xlwt
import linecache
import sys
from math import floor
import pandas as pd
import datetime

#Date_validation----------------------------------------------------------------------------------------------------------------

def fn_date__(str_, **kwargs):
    
    temp_date = re.sub(r'[^\w]', ' ', str_).strip()
    temp_date = re.sub('\s+', '', temp_date)
    
    # temp_date = re.sub(r'O|Q', '0', temp_date)
    # temp_date = re.sub(r'B', '8', temp_date)
    # temp_date = re.sub(r'l', '1', temp_date)
    # temp_date = re.sub(r'J|\?', '7', temp_date)
    
    # temp_date = temp_date.lower().replace("rd", "").replace("nd", "").replace("st", "").replace("th", "")
        
    # temp_date = temp_date.replace('january', '01').replace('february', '02').replace('march', '03').replace('april', '04').replace('may', '05').replace('june', '06')
    # temp_date = temp_date.replace('july', '07').replace('august', '08').replace('september', '09').replace('october', '10').replace('november', '11').replace('december', '12')

    # temp_date = temp_date.replace('jan', '01').replace('feb', '02').replace('mar', '03').replace('apr', '04').replace('may', '05').replace('jun', '06')
    # temp_date = temp_date.replace('jul', '07').replace('aug', '08').replace('sep', '09').replace('oct', '10').replace('nov', '11').replace('dec', '12')    
    
    # temp_date = temp_date.replace('jan', 'Jan').replace('feb', 'Feb').replace('mar', 'Mar').replace('apr', 'Apr').replace('may', 'May').replace('jun', 'Jun')
    # temp_date = temp_date.replace('jul', 'Jul').replace('aug', 'Aug').replace('sep', 'Sep').replace('oct', 'Oct').replace('nov', 'Nov').replace('dec', 'Dec')

    try:
        date_ = datetime.datetime.strptime(temp_date, "%d%m%y")
        final_date = date_.strftime("%d/%m/%y")      
    except:
        try: # "%d %b %Y"
            date_ = datetime.datetime.strptime(temp_date, "%d%b%Y")
            final_date = date_.strftime("%d/%m/%y") 
        except:
            try:
                date_ = datetime.datetime.strptime(temp_date, "%d%b%y")
                final_date = date_.strftime("%d/%m/%y") 
                
            except:
                try:
                    date_ = datetime.datetime.strptime(temp_date, "%d%m%Y")
                    final_date = date_.strftime("%d/%m/%y") 
                    
                except:
                    try:
                        date_ = datetime.datetime.strptime(temp_date, "%B%d%Y")
                        final_date = date_.strftime("%d/%m/%y") 
                        
                    except:
                        try:
                            date_ = datetime.datetime.strptime(temp_date, "%m%d%Y")
                            final_date = date_.strftime("%d/%m/%y") 
                            
                        except:
                            try:
                                date_ = datetime.datetime.strptime(temp_date, "%b%d%Y")
                                final_date = date_.strftime("%d/%m/%y") 
                                
                            except:
                                try:
                                    date_ = datetime.datetime.strptime(temp_date, "%d%B%Y")
                                    final_date = date_.strftime("%d/%m/%y") 
                                except:
                                    try:
                                        date_ = datetime.datetime.strptime(temp_date, "%b%d%Y%I%M%p")
                                        final_date = date_.strftime("%d/%m/%y") 
                                    except:
                                        return temp_date
                                                                                  
    return final_date

class swift_parser():

    def __init__(self, path, fnames, header_keyword_file, trans_keyword_file, trans_identifiers_file, trans_out_file):

        self.path = path
        self.fnames = fnames
        self.header_keyword_file = header_keyword_file
        self.trans_keyword_file = trans_keyword_file
        self.trans_identifiers_file = trans_identifiers_file
        self.trans_out_file = trans_out_file
        
        #read swift message and clean it
                
        with open(self.path + fnames[1]) as f:    
            self.swift = f.read()
            f.close()
            
        self.swift = re.sub(r' +', ' ', self.swift)
        self.swift = re.sub(r'\t+', ' ', self.swift)
        self.swift.strip()

        nls = self.swift.split('\n')
        nls_stripped = []

        for nls_ in nls:
            nls_stripped.append(nls_.strip())

        self.swift = '\n'.join(nls_stripped)
        
        p = re.compile(r' no\. ', re.I)
        self.swift = re.sub(p, ' no ', self.swift)

        p = re.compile(r'p\.a\.', re.I)
        self.swift = re.sub(p, 'p.a.,', self.swift)
        
        p = re.compile(r'INR\.', re.I)
        self.swift = re.sub(p, 'INR ', self.swift)
        
        p = re.compile(r'n\.a\.', re.I)
        self.swift = re.sub(p, 'n.a.,', self.swift)
        
        #read swift message template and clean it
        
        with open(self.path + fnames[0]) as f:
            self.swift_template = f.read()
            
        self.swift_template = re.sub(r' +', ' ', self.swift_template)
        self.swift_template = re.sub(r'\t+', ' ', self.swift_template)
        self.swift_template.strip()

        nls = self.swift_template.split('\n')
        self.st_nls_stripped = []

        for nls_ in nls:
            self.st_nls_stripped.append(nls_.strip())

        self.swift_template = '\n'.join(self.st_nls_stripped)
    
    def extract_header_values(self):
        
        #read keywords

        with open(self.path + self.header_keyword_file) as f:
            reader = csv.reader(f)
            self.h_keywords = [r for r in reader]
            self.h_keywords.pop(0) # remove header

        #identify header text.
        #The position before the clause marks the end of the last transaction.
        #We use a customized version of jaccard similarity to identify the clause present.

        n = 0 #end of header text not yet identified

        #split the swift message into sentences
        header_sents = PunktSentenceTokenizer().tokenize(self.swift)
        self.swift_template_sents = PunktSentenceTokenizer().tokenize(self.swift_template)
        
        #loop through each sentence of the string to identify the sentience in which the header text ends.
        for h_sent in header_sents:
            if n==0:
                tokenizer = RegexpTokenizer(r'\w+')
                h_sent_words = tokenizer.tokenize(h_sent.lower())
                h_sent_words = list(bigrams(h_sent_words))

                #loop through each sentence of the template to identify the clause that marks the end of the header text.
                for sft_sent in self.swift_template_sents:
                    if n==0:

                        sft_sent_ = re.sub(r'\[.*?\]', '', sft_sent)
                        sft_sent_ = re.sub(r'Option\d\*\*\*', '', sft_sent_)
                                
                        tokenizer = RegexpTokenizer(r'\w+')
                        sft_sent_words = tokenizer.tokenize(sft_sent_.lower())
                        sft_sent_bgs = list(bigrams(sft_sent_words))
                        
                        common_words, similarity_ratio = self.jaccard_similarity(h_sent_words, sft_sent_bgs, rigid = False)

                        if similarity_ratio > 0.70 or (len(sft_sent_words) < 20 and similarity_ratio > 0.49): #similarity threshold set to 97.5%

                            #once the sentence is identified, search of the first common word in the clause and sentence to mark the end position of the header text.
                            for word in sft_sent_bgs:
                                if word in common_words:
                                    header_endpos = h_sent.lower().find(word[0])
                                    h_sent_words_index = header_sents.index(h_sent)

                                    #slice the header text at its end
                                    h_sent_correct = h_sent[:header_endpos]
                                    n = 1 #header text identified. Exit all for loops
                                    break

        #reformulate the header text so that it contains only till the actual end of it.
        header_sents = header_sents[:h_sent_words_index]
        header_sents.append(h_sent_correct)
        header_text = ''.join(header_sents)

        #save the end position of the header section as these areas should not be considered for difference later.
        self.header_end = len(header_text)

        self.headervalues = {}
        self.headervalues['header_text'] = header_text

        extracted_values = {}
        keyword_indices = []
        unfound_values = []
        
        #loop through each set of keywords for a particular field
        for kws in self.h_keywords:
            #keyword not yet found
            n = 0
            for kw in kws:
                if n == 0 and kw:
                    kwtf = kw.replace('[','').replace(']','')
                    kwtf = re.sub(r' +', r'.*?', kwtf)
                    kwtf = re.compile(kwtf, re.I)
                    for m in kwtf.finditer(header_text):
                        keyword_indices.append([kw, m.start(), m.end()])
                        n = 1 #keyword found. Exit loop
                        break
            if n == 0: #if no keyword for a particular field is found, add it to unfoud values
                kws_temp = []
                for kw in kws:
                    if kw:
                        kws_temp.append(kw)

                unfound_values.append(kws_temp)

        #sort keywords based on start position in the transaction text
        keyword_indices = sorted(keyword_indices, key=lambda x: x[1])
        
        #loop through all identified keywords
        for kwi in keyword_indices:
            
            if kwi[0].find('[') != -1:
                fromval = kwi[1]
            else:
                fromval = kwi[2]

            if keyword_indices.index(kwi) + 1 == len(keyword_indices):
                #if lst keyword, slice values between end of the keyword and the end of the transaction text. Also validate it.
                ex_val = self.validations(header_text[fromval:], kwi[0])
            else:
                #slice values between end of the keyword and the start of the next keyword. Also validate them.
                ex_val = self.validations(header_text[fromval:keyword_indices[keyword_indices.index(kwi)+1][1]], kwi[0])

            #for a dictionary which has the extracted values as key value pair with field name as key an extracted value as value
            extracted_values[kwi[0]] = ex_val

        #add the extracted values to the tranaction in the transactions dictionary. Now the transactions dictionary will have the full transaction text
        #and the extracted values for each transaction.
        self.headervalues['extracted_values'] = extracted_values
        self.headervalues['unfound_values'] = unfound_values

    def extract_transaction_values(self):

        
        with open(self.path + self.trans_identifiers_file) as f:
            reader = csv.reader(f)
            self.tis = [r for r in reader][0]
            
        #read keywords
        
        with open(self.path + self.trans_keyword_file) as f:
            reader = csv.reader(f)
            self.trans_keywords = [r for r in reader]
            self.trans_keywords.pop(0) # remove header

        #identify number of transactions
        self.trans_pos = []

        for ti in self.tis:
            
            titf = ti.replace('[','').replace(']','')
            titf = re.sub(r' +', r'.*?', titf)

            swift_temp = self.swift.replace('\n',' ')

            p = re.compile(titf, re.I)
            for m in p.finditer(swift_temp):
                if not any((m.start() - i) < 400 for i in self.trans_pos):
                    self.trans_pos.append(m.start())
        
        for i in reversed(self.trans_pos):
            if i < self.header_end:
                self.trans_pos.remove(i)
                
        #split_transactions into a dictionary
        n = 0 #last transaction not yet identified
        self.transactions = {}

        for tp in self.trans_pos:

            #handle last transaction differently as there is no defined end position.
            if self.trans_pos.index(tp) == len(self.trans_pos) -1:

                #consider swift message from start position of last trans
                last_trans = self.swift[tp:]

                #split the last line into sentences and identify in which sentence is an actual clause present.
                # self.sent_tokenizer = PunktSentenceTokenizer().tokenize

                last_trans_sents = PunktSentenceTokenizer().tokenize(last_trans)

                #The position before the clause marks the end of the last transaction.
                #We use a customized version of jaccard similarity to identify the clause present.

                #loop through each sentence of the string to identify the sentience in which the last transaction ends.
                for lt_sent in last_trans_sents:
                    if n==0:
                        tokenizer = RegexpTokenizer(r'\w+')
                        lt_sent_words = tokenizer.tokenize(lt_sent.lower())
                        lt_sent_words = list(bigrams(lt_sent_words))

                        #loop through each sentence of the template to identify the clause that marks the end of the last transaction.
                        for sft_sent in self.swift_template_sents:
                            if n==0:

                                sft_sent_ = re.sub(r'\[.*?\]', '', sft_sent)
                                sft_sent_ = re.sub(r'Option\d\*\*\*', '', sft_sent_)
                                
                                tokenizer = RegexpTokenizer(r'\w+')
                                sft_sent_words = tokenizer.tokenize(sft_sent_.lower())
                                sft_sent_bgs = list(bigrams(sft_sent_words))

                                common_words, similarity_ratio = self.jaccard_similarity(lt_sent_words, sft_sent_bgs, rigid = False)

                                if similarity_ratio > 0.70 or (len(sft_sent_words) < 20 and similarity_ratio > 0.49): #similarity threshold set to 97.5%

                                    #once the sentence is identified, search of the first common word in the clause and sentence to mark the end position of the last transaction
                                    for word in sft_sent_bgs:
                                        if word in common_words:
                                            last_trans_endpos = lt_sent.lower().find(word[0])
                                            lt_sent_index = last_trans_sents.index(lt_sent)

                                            #slice the last transaction at its end
                                            lt_sent_correct = lt_sent[:last_trans_endpos]
                                            n = 1 #last transaction identified. Exit all for loops
                                            break

                #reformulate the last transaction so that it contains only till the actual end of it.
                last_trans_sents = last_trans_sents[:lt_sent_index]
                last_trans_sents.append(lt_sent_correct)
                last_trans_actual = ''.join(last_trans_sents)

                #append the last transaction to the transactions dictionary.
                self.transactions['TRANSACTION_' + str(len(self.trans_pos))] = {}
                self.transactions['TRANSACTION_' + str(len(self.trans_pos))]['text'] = last_trans_actual

                #save the start and end positions of the whole transaction section as these areas should not be considered for difference later.
                self.trans_start = self.trans_pos[0]
                self.trans_end = self.trans_pos[-1] + len(last_trans_actual)

            else:

                #extract other transactions based on start and end positions and append the last transaction to thw transactions dictionary.
                self.transactions['TRANSACTION_' + str(self.trans_pos.index(tp)+1)] = {}
                self.transactions['TRANSACTION_' + str(self.trans_pos.index(tp)+1)]['text'] = self.swift[tp:self.trans_pos[self.trans_pos.index(tp)+1]]

        #extract values from each transaction based on keywords
        for key, value in self.transactions.items():
            extracted_values = {}
            keyword_indices = []
            unfound_values = []
            
            #loop through each set of keywords for a particular field
            for kws in self.trans_keywords:
                #keyword not yet found
                n = 0
                for kw in kws:
                    if n == 0 and kw:
                        kwtf = re.sub(r' +', r'.*?', kw)
                        kwtf = re.compile(kwtf, re.I)
                        for m in kwtf.finditer(value['text']):
                            keyword_indices.append([kw, m.start(), m.end()])
                            n = 1 #keyword found. Exit loop
                            break
                if n == 0: #if no keyword for a particular field is found, add it to unfoud values
                    kws_temp = []
                    for kw in kws:
                        if kw:
                            kws_temp.append(kw)

                    unfound_values.append(kws_temp)

            #sort keywords based on start position in the transaction text
            keyword_indices = sorted(keyword_indices, key=lambda x: x[1])
            
            #loop through all identified keywords
            for kwi in keyword_indices:

                if keyword_indices.index(kwi) + 1 == len(keyword_indices):
                    #if lst keyword, slice values between end of the keyword and the end of the transaction text. Also validate it.
                    ex_val = self.validations(value['text'][kwi[2]:], kwi[0])
                else:
                    #slice values between end of the keyword and the start of the next keyword. Also validate them.
                    ex_val = self.validations(value['text'][kwi[2]:keyword_indices[keyword_indices.index(kwi)+1][1]], kwi[0])

                #for a dictionary which has the extracted values as key value pair with field name as key an extracted value as value
                extracted_values[kwi[0]] = ex_val

            #add the extracted values to the tranaction in the transactions dictionary. Now the transactions dictionary will have the full transaction text
            #and the extracted values for each transaction.
            self.transactions[key]['extracted_values'] = extracted_values
            self.transactions[key]['unfound_values'] = unfound_values

        with open(self.path + self.trans_out_file) as f:
            reader = csv.reader(f)
            self.t_out = [r for r in reader]
            
        headers = self.t_out[0]
        header_variances = self.t_out[1]
        all_trans_val = []

        for transaction, values in self.transactions.items():

            trans_vals = []
            # trans_vals.append(transaction)
            ex_vals = values['extracted_values']

            for header in headers:
                
                if header_variances[headers.index(header)]:
                    h_vars = header_variances[headers.index(header)]
                    h_vars = h_vars.split('/')               

                    n = 0

                    for key, val in ex_vals.items():
                        if not n:
                            for h_v in h_vars:
                                h_v = re.sub(r' +', r'.*?', h_v)
                                if re.findall(h_v, key, re.I):
                                    for i in range(2, len(self.t_out)):
                                        if not self.t_out[i][headers.index(header)]:
                                            break
                                        search_pat = self.t_out[i][headers.index(header)]
                                        if search_pat.find('|') > -1:
                                            pat = search_pat.split('|')[0]
                                            sub_val = search_pat.split('|')[1]
                                        elif search_pat.find('fn_') > -1:
                                            try:
                                                fn_call = search_pat + "('" + val + "')"
                                                val = eval(fn_call)
                                            except Exception as e:
                                                print(str(e))
                                            n =1
                                            trans_vals.append(val)
                                            break
                                        else:
                                            pat = re.compile(search_pat, re.I)
                                            sub_val = ''

                                        try:
                                            val = re.findall(pat, val)[0]
                                            if val and sub_val:
                                                val = sub_val
                                            n =1
                                            trans_vals.append(val)
                                            break
                                        except:
                                            continue
                                            
                                    if not n:
                                        trans_vals.append(val)
                                        n =1
                                    break
                        else:
                            break

                    if not n:
                        trans_vals.append('Not found')
                else:
                    trans_vals.append('')

            all_trans_val.append(trans_vals)

        # headers.insert(0, 'Transaction Number')
        all_trans_val.insert(0,headers)

        head_style = xlwt.XFStyle()

        font = xlwt.Font()
        pattern = xlwt.Pattern()
        font.bold = True
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['pale_blue']
        head_style.font = font
        head_style.pattern = pattern

        no_value_style = xlwt.XFStyle()
        nv_pattern = xlwt.Pattern()
        nv_pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        nv_pattern.pattern_fore_colour = xlwt.Style.colour_map['light_yellow']
        no_value_style.pattern = nv_pattern

        xldoc = xlwt.Workbook()
        sheet1 = xldoc.add_sheet("Sheet1", cell_overwrite_ok=True)
        for i, row in enumerate(all_trans_val):
            for j, col in enumerate(row):
                if i == 0:
                    sheet1.write(i, j, col, style=head_style)
                elif col == 'Not found':
                    sheet1.write(i, j, col, style=no_value_style)
                else:
                    sheet1.write(i, j, col)

        xldoc.save(self.path  + 'transactions.xls')
            
    def validations(self, input_text, keyword):
        
        #split the text to be validated based on new lines
        input_text_lines = input_text.split('\n')
        input_text_valid = []

        #append valid text to the above defined list
        for line in input_text_lines:

            #line to contain atleast one letter or number. It should not contain the word narrative
            if re.findall(r'[a-zA-Z0-9]+', line) and line.lower().find(r'narrative') == -1:
                input_text_valid.append(line)

        #join all valid lines with spaces. This makes all multiline values into single line values         
        input_text = ' '.join(input_text_valid)
        
        #consider the value only from the first letter or number
        input_text = str(re.findall(r'[^a-zA-Z0-9]*(.*)', input_text)[0])

        #dates can contain only letters, numbers, period, hyphen, forward / backward slashes
        if keyword.lower().find('date') != -1:
            input_text_valid = [str(i) for i in re.findall(r'[[A-z]*[0-9]*,*\.*\-*/*]*',input_text) if str(i) != ''] #,*\.*\-*/*
            input_text = ''.join(input_text_valid)

        return input_text

    def generate_output(self):
        
        left_clause_bt = ''
        right_clause_bt = ''
        left_clause_at = ''
        right_clause_at = ''
        
        left_header_text = self.headervalues['header_text']

        h_extracted_values = json.dumps(self.headervalues['extracted_values'])

        h_extracted_values_df = pd.DataFrame(self.headervalues['extracted_values'], index=[0])
        pd.set_option('max_colwidth', -1)
        h_extracted_values_df = h_extracted_values_df.transpose()
        h_extracted_values_df.reset_index(inplace=True)
        h_extracted_values_df.columns = ['FIELD NAME', 'VALUE']
        h_extracted_values_html = h_extracted_values_df.to_html(index=None)
        h_extracted_values_html = h_extracted_values_html.replace('style="text-align: right;','style="text-align: center;')

        # h_extracted_values = h_extracted_values.replace('{','').replace('}','')
        # h_extracted_values = h_extracted_values.replace('", "','"<br><br>"').replace(':',' : ')

        right_header_text = h_extracted_values_html + '<br><br>Missing Fields: ' + str(self.headervalues['unfound_values'])

        for sent in self.final_diff[0]:
            if sent[0] != 'NA':
                left_clause_bt = left_clause_bt + sent[0].lower() + '<br><br><br>'

        for sent in self.final_diff[0]:
            if sent[1] != 'NA':
                right_clause_bt = right_clause_bt + sent[1].lower() + '<br><br><br>'
        
        for sent in self.final_diff[1]:
            if sent[0] != 'NA':
                left_clause_at = left_clause_at + sent[0].lower() + '<br><br><br>'

        for sent in self.final_diff[1]:
            if sent[1] != 'NA':
                right_clause_at = right_clause_at + sent[1].lower() + '<br><br><br>'

        symbols = [['###', 'value'],['\+\+\+', 'addition'],['---', 'negation']]

        for symbol in symbols:

            symbol_indices = []

            pattern = re.compile(symbol[0], re.I)

            for m in pattern.finditer(left_clause_bt):
                symbol_indices.append([m.start(), m.end()])

            for hi in symbol_indices:

                if symbol_indices.index(hi) % 2 == 0:
                    left_clause_bt = left_clause_bt[:hi[0]] + 'O' + symbol[0][1] + 'T' + left_clause_bt[hi[0]+3:]
                else:
                    left_clause_bt = left_clause_bt[:hi[0]] + 'C' + symbol[0][1] + 'T' + left_clause_bt[hi[0]+3:]

            symbol_indices = []

            for m in pattern.finditer(left_clause_at):
                symbol_indices.append([m.start(), m.end()])

            for hi in symbol_indices:

                if symbol_indices.index(hi) % 2 == 0:
                    left_clause_at = left_clause_at[:hi[0]] + 'O' + symbol[0][1] + 'T' + left_clause_at[hi[0]+3:]
                else:
                    left_clause_at = left_clause_at[:hi[0]] + 'C' + symbol[0][1] + 'T' + left_clause_at[hi[0]+3:]

            symbol_indices = []

            for m in pattern.finditer(right_clause_bt):
                symbol_indices.append([m.start(), m.end()])

            for hi in symbol_indices:

                if symbol_indices.index(hi) % 2 == 0:
                    right_clause_bt = right_clause_bt[:hi[0]] + 'O' + symbol[0][1] + 'T' + right_clause_bt[hi[0]+3:]
                else:
                    right_clause_bt = right_clause_bt[:hi[0]] + 'C' + symbol[0][1] + 'T' + right_clause_bt[hi[0]+3:]

            symbol_indices = []

            for m in pattern.finditer(right_clause_at):
                symbol_indices.append([m.start(), m.end()])

            for hi in symbol_indices:

                if symbol_indices.index(hi) % 2 == 0:
                    right_clause_at = right_clause_at[:hi[0]] + 'O' + symbol[0][1] + 'T' + right_clause_at[hi[0]+3:]
                else:
                    right_clause_at = right_clause_at[:hi[0]] + 'C' + symbol[0][1] + 'T' + right_clause_at[hi[0]+3:]

        left_clause_bt = left_clause_bt.replace('O#T', '<span class="value">').replace('C#T', '</span>')
        left_clause_bt = left_clause_bt.replace('O+T', '<span class="addition">').replace('C+T', '</span>')
        left_clause_bt = left_clause_bt.replace('O-T', '<span class="negation">').replace('C-T', '</span>')

        left_clause_at = left_clause_at.replace('O#T', '<span class="value">').replace('C#T', '</span>')
        left_clause_at = left_clause_at.replace('O+T', '<span class="addition">').replace('C+T', '</span>')
        left_clause_at = left_clause_at.replace('O-T', '<span class="negation">').replace('C-T', '</span>')        

        right_clause_bt = right_clause_bt.replace('O#T', '<span class="value">').replace('C#T', '</span>')
        right_clause_bt = right_clause_bt.replace('O+T', '<span class="addition">').replace('C+T', '</span>')
        right_clause_bt = right_clause_bt.replace('O-T', '<span class="negation">').replace('C-T', '</span>')
             
        right_clause_at = right_clause_at.replace('O#T', '<span class="value">').replace('C#T', '</span>')
        right_clause_at = right_clause_at.replace('O+T', '<span class="addition">').replace('C+T', '</span>')
        right_clause_at = right_clause_at.replace('O-T', '<span class="negation">').replace('C-T', '</span>')

        #incorrect css class check

        span_start_indices = []
        span_end_indices = []

        symbols = ['<span class="value">', '<span class="addition">']

        for si in symbols:
            pattern = re.compile(si, re.I)

            for m in pattern.finditer(right_clause_bt):
                span_start_indices.append([m.start(), m.end()])

        pattern = re.compile('</span>', re.I)

        for m in pattern.finditer(right_clause_bt):
            span_end_indices.append([m.start(), m.end()])

        span_start_indices = sorted(span_start_indices)
        span_end_indices = sorted(span_end_indices)

        remove_indices = []

        if len(span_start_indices) != len(span_end_indices):
            
            for i in range(len(span_start_indices)):
                
                try:
                    if not (span_start_indices[i][1] <= span_end_indices[i][0] <=  span_end_indices[i][1] <= span_start_indices[i+1][0]):
                        remove_indices.append(span_start_indices[i])
                except IndexError as e:
                    remove_indices.append(span_start_indices[i])
       
        for ri in reversed(remove_indices):
            right_clause_bt_1 = right_clause_bt[:ri[0]]
            right_clause_bt_2 = right_clause_bt[ri[1]+1:]

            right_clause_bt = right_clause_bt_1 + right_clause_bt_2

        span_start_indices = []
        span_end_indices = []

        symbols = ['<span class="value">', '<span class="addition">']

        for si in symbols:
            pattern = re.compile(si, re.I)

            for m in pattern.finditer(right_clause_at):
                span_start_indices.append([m.start(), m.end()])

        pattern = re.compile('</span>', re.I)

        for m in pattern.finditer(right_clause_at):
            span_end_indices.append([m.start(), m.end()])

        span_start_indices = sorted(span_start_indices)
        span_end_indices = sorted(span_end_indices)

        remove_indices = []

        if len(span_start_indices) != len(span_end_indices):
            
            for i in range(len(span_start_indices)):
                
                try:
                    if not (span_start_indices[i][1] <= span_end_indices[i][0] <=  span_end_indices[i][1] <= span_start_indices[i+1][0]):
                        remove_indices.append(span_start_indices[i])
                except IndexError as e:
                    remove_indices.append(span_start_indices[i])
        
        for ri in reversed(remove_indices):
            right_clause_at_1 = right_clause_at[:ri[0]]
            right_clause_at_2 = right_clause_at[ri[1]+1:]

            right_clause_at = right_clause_at_1 + right_clause_at_2

        lt_text = ''
        rt_text = ''

        n = 0

        for key, values in self.transactions.items():
            
            if n == len(self.transactions):
                lt_text = lt_text + values['text'] + '<br><br>'
            else:
                lt_text = lt_text + values['text']

            extracted_values = json.dumps(values['extracted_values'])

            extracted_values_df = pd.DataFrame(values['extracted_values'], index=[0])
            pd.set_option('max_colwidth', -1)
            extracted_values_df = extracted_values_df.transpose()
            extracted_values_df.reset_index(inplace=True)
            extracted_values_df.columns = ['FIELD NAME', 'VALUE']
            extracted_values_html = extracted_values_df.to_html(index=None)
            extracted_values_html = extracted_values_html.replace('style="text-align: right;','style="text-align: center;')
            
            extracted_values = extracted_values.replace('{','').replace('}','')
            extracted_values = extracted_values.replace('", "','"<br><br>"').replace(':',' : ')

            unfound_values = str(values['unfound_values'])

            if n == 0:
                rt_text = rt_text + '<b>' + key + '</b><br><br>' + extracted_values_html + '<br><br>' + 'Missing Field(s): ' + unfound_values  + '<br><br>'
            else:
                rt_text = rt_text + '<b>' + key + '</b><br><br>' + extracted_values_html + '<br><br>' +   'Missing Field(s): ' + unfound_values
        
        with open(self.path + 'html\\layout.txt') as f:
            html_str = f.read()

        html_str = html_str.replace('***header_text***', left_header_text.replace('\n', '<br>'))
        html_str = html_str.replace('***header_vals***', str(right_header_text))
        html_str = html_str.replace('***cbt_left***', left_clause_bt)
        html_str = html_str.replace('***cbt_right***', right_clause_bt)
        html_str = html_str.replace('***t_left***', lt_text.replace('\n', '<br>'))
        html_str = html_str.replace('***t_right***', rt_text)
        html_str = html_str.replace('***cat_left***', left_clause_at)
        html_str = html_str.replace('***cat_right***', right_clause_at)

        with open(os.path.join(self.path, 'html\\output.html'), "w") as file:
            file.write(html_str)

    def mark_difference(self):

        self.clause_values = {}
        
        simalirity_map = self.gen_similarity_map()

        sentence_indices = [[], []]

        n = 1

        for sm in simalirity_map:
            for key, value in sm.items():
                swift_sent_index = key
                if value == 'NA':
                    if not swift_sent_index + 1/1000 in list(sm.keys()):
                        temp_sent_index = 999999999
                    else:
                        temp_sent_index =  list(sm[swift_sent_index + 1/1000].keys())[0]
                else:
                    temp_sent_index = list(value.keys())[0]
                
                if n:
                    sentence_indices[0].append([temp_sent_index, swift_sent_index])
                else:
                    sentence_indices[1].append([temp_sent_index, swift_sent_index])

            n = 0

        sentence_indices[0] = sorted(sentence_indices[0], key=lambda x: x[1])
        sentence_indices[1] = sorted(sentence_indices[1], key=lambda x: x[1])

        self.final_diff = [[], []]

        n = 0

        temp_sents_register = []

        for sis in sentence_indices:
            
            for si in sis:

                temp_sents_register.append(si[0])

                if (sis.index(si) != 0) and (sis[sis.index(si)-1][0] == si[0] or int(floor(sis[sis.index(si)-1][1])) == int(floor(si[1]))):
                    continue
                else:

                    swift_sent = self.swift_sentences[n][int(floor(si[1]))]
                    
                    if (sis.index(si) + 1) != len(sis) and sis[sis.index(si)+1][0] == si[0] and int(floor(sis[sis.index(si)+1][1])) != int(floor(si[1])):
                        
                        for sis_next in range(sis.index(si)+1, len(sis)):
                            
                            current_sis = sis[sis_next]

                            if current_sis[0] == si[0]:
                                swift_sent = swift_sent + ' ' + self.swift_sentences[n][int(floor(current_sis[1]))]
                            else:
                                break

                    if si[0] == 999999999:
                        swift_sent_temp = '+++' + swift_sent + '+++'
                        template_sent_temp = 'NA'
                    else:
                            
                        template_sent = self.swift_template_sents[si[0]]

                        if (sis.index(si) + 1) != len(sis) and int(floor(sis[sis.index(si)+1][1])) == int(floor(si[1])):

                            template_sent = ''
                            sents_w_indices = [[si[0], self.swift_template_sents[si[0]]]]

                            for sis_next in range(sis.index(si)+1, len(sis)):
                                
                                current_sis = sis[sis_next]

                                if int(floor(current_sis[1])) == int(floor(si[1])):
                                    if current_sis[0] != si[0]:
                                        sents_w_indices.append([current_sis[0], self.swift_template_sents[current_sis[0]]])
                                else:
                                    break

                            sents_w_indices = sorted(sents_w_indices, key=lambda x: x[0])

                            for swi in sents_w_indices:
                                template_sent = template_sent + ' ' + swi[1]


                        swift_sent_temp = swift_sent
                        template_sent_temp = template_sent

                        template_sent_temp, swift_sent_temp  = self.gen_diff(template_sent_temp, swift_sent_temp)

                    self.final_diff[n].append([template_sent_temp, swift_sent_temp])

            n = 1
        
        temp_sents_register = list(set(temp_sents_register))

        if not len(temp_sents_register) == len(self.swift_template_sents):

            temp_sents_compare = [i for i in range(0, len(self.swift_template_sents))]
            n = 0

            for i in temp_sents_compare:
                if not temp_sents_compare[i] in temp_sents_register:
                    
                    sent = self.swift_template_sents[i]

                    if n == 0:
                        if re.findall(r'Option\d+\*\*\*', sent, re.I):
                            n = 1
                            continue
                    
                    sent = '---' + sent + '---'
                    self.final_diff[1].append([sent, 'NA'])

    def diff(self, v1, v2):

        v1 = v1.lower()
        v2 = v2.lower()

        # ll = [[word_tokenize(w), ' '] for w in v1.split()]
        # v1_words = list(chain(*list( chain(*ll))))

        # ll = [[word_tokenize(w), ' '] for w in v2.split()]
        # v2_words = list(chain(*list( chain(*ll))))

        v1_words = word_tokenize(v1)
        v2_words = word_tokenize(v2)

        # v1_words = list(v1)
        # v2_words = list(v2)

        clubbed_diff = difflib.ndiff(v1_words, v2_words)

        lost_diff = []
        gained_diff = []

        for diff_word in clubbed_diff:
            
            if diff_word.find('?') == 0:
                continue

            if diff_word.find('-') == 0:
                word = diff_word.replace('- ','---',1)
                word = word + '---'
                word = re.sub(r' +', ' ',word)
                word = word.strip()
                lost_diff.append(word)

            elif diff_word.find('+') == 0:
                word = diff_word.replace('+ ','+++',1)
                word = word + '+++'
                word = re.sub(r' +', ' ',word)
                word = word.strip()
                gained_diff.append(word)
            else:
                word = re.sub(r' +', ' ',diff_word)
                word = word.strip()
                lost_diff.append(word)
                gained_diff.append(word)

        v1 = ' '.join(lost_diff)
        v2 = ' '.join(gained_diff)

        return v1, v2

    def gen_diff(self, sent1, sent2):
        
        sent1 = re.sub(r' +', ' ', sent1)
        sent1 = re.sub(r'\t+', ' ', sent1)
        sent1 = re.sub(r'\n+', ' ', sent1)

        sent2 = re.sub(r' +', ' ', sent2)
        sent2 = re.sub(r'\t+', ' ', sent2)
        sent2 = re.sub(r'\n+', ' ', sent2)

        sent1 = sent1.strip()
        sent2 = sent2.strip()

        lost_text , gained_text = self.diff(sent1, sent2)

        # lost_text = lost_text.replace('--- ---',' ')
        # gained_text = gained_text.replace('+++ +++',' ')

        lost_text = re.sub(r'\-\-\-option\d\*\*\*\-\-\-', '', lost_text)
        lost_text = lost_text.replace('---[','###[').replace(']---',']###')

        if lost_text.find('###[') != -1:
            lost_text, gained_text = self.extract_clause_values(lost_text, gained_text)

        return lost_text, gained_text

    def extract_clause_values(self, lost_text, gained_text):
        
        value_pats = lost_text.split('###[')   

        for value in value_pats:
            
            if value_pats.index(value) + 1 != len(value_pats):

                prev_words = value[-20:]
                prev_words = prev_words.strip()

                field_name = value_pats[value_pats.index(value) + 1]
                field_name = field_name.split(']###')[0]

                post_words = value_pats[value_pats.index(value) + 1]
                post_words = post_words.split(']###')[1]

                post_words = post_words[:20]
                post_words = post_words.strip()

                if len(post_words) < 2:
                    post_words = ''

                prev_words = prev_words.replace('(','\(').replace(')','\)').replace('.','\.').replace('?','\?').replace('*','\*')
                post_words = post_words.replace('(','\(').replace(')','\)').replace('.','\.').replace('?','\?').replace('*','\*')
                
                if prev_words.find('###') > -1:
                    prev_words = prev_words.split('###')[-1]

                if post_words.find('###') > -1:
                    post_words = post_words.split('###')[0]

                to_find_pat = prev_words + '(.*?)' + post_words
                if not post_words:
                    to_find_pat = to_find_pat + '$'

                to_find_pat = re.sub(r' +', r'.*?',to_find_pat)

                if re.findall(to_find_pat, gained_text, re.IGNORECASE):
                    ph_val = re.findall(to_find_pat, gained_text, re.IGNORECASE)[0]

                    ph_val_actual = ph_val.replace('+++',' ')
                    ph_val_actual = '###' + ph_val_actual + '###'
                    gained_text = gained_text.replace(ph_val, ph_val_actual)

                else:

                    ph_val_actual = 'Not found'
            
                field_name_actual = field_name.replace('---','')
                lost_text = lost_text.replace(field_name, field_name_actual)

                gained_text = re.sub(r' +', ' ', gained_text)
                gained_text = re.sub(r'\t+', ' ', gained_text)
                gained_text = re.sub(r'\n+', ' ', gained_text)

                self.clause_values[field_name_actual] = ph_val_actual.replace('###','')

        return lost_text, gained_text

    def gen_similarity_map(self):
        
        swift_bt = self.swift[self.header_end:self.trans_start]
        swift_at = self.swift[self.trans_end:]
        
        self.swift_sentences = []

        self.swift_sentences.append(PunktSentenceTokenizer().tokenize(swift_bt))
        self.swift_sentences.append(PunktSentenceTokenizer().tokenize(swift_at))

        similarity_map = []

        for sents in self.swift_sentences:
            for sent in reversed(sents):
                if not re.findall(r'[a-zA-Z0-9]+', sent):
                    sents.remove(sent)

            similarity_map.append(self.map_sentences(sents, self.swift_template_sents))

        sts_indices = []

        for s_map in similarity_map:
            for _,val in s_map.items():
                if val != 'NA':
                    for key in val.keys():
                        sts_indices.append(key)

        sts_indices = sorted(sts_indices)

        recheck_st_sents = []
        recheck_st_sents_indices = []

        for i in range(len(self.swift_template_sents)):
            if i not in sts_indices:
                if self.swift_template_sents[i].lower().find('option') == -1 and self.swift_template_sents[i].lower().find('***') == -1:
                    recheck_st_sents.append(self.swift_template_sents[i])
                    recheck_st_sents_indices.append(i)

        inverse_similarity_map = []
        
        for sents in self.swift_sentences:
            inverse_similarity_map.append(self.map_sentences(recheck_st_sents, sents, inverse=1))

        for s_map in inverse_similarity_map:
            n = 1
            old_k = 0
            for key,val in s_map.items():
                if val != 'NA': 
                    for k, v in val.items():
                        if old_k < k:
                            n = 1
                        old_k = k
                        st_index = k + (n / 1000)
                        matched_sent = {recheck_st_sents_indices[key] : v}
                        n+= 1

                    similarity_map[inverse_similarity_map.index(s_map)][st_index] = matched_sent

        return similarity_map
            
    def map_sentences(self, swift_sentences, template_sentences, inverse = 0):
        
        currency_list = ['USD', 'GBP', 'SGD', 'MYR', 'INR']
        
        similarity_mapping = {}

        for sft_sent in swift_sentences:

            similarity_mapping[swift_sentences.index(sft_sent)] = {}

            tokenizer = RegexpTokenizer(r'\w+')
            sft_sent_words = tokenizer.tokenize(sft_sent.lower())
            sft_sent_bgs = list(bigrams(sft_sent_words))

            n = 0

            opt_sim_ratio_values = []

            curr_check = 1

            for sft_template_sent in template_sentences:
                tokenizer = RegexpTokenizer(r'\w+')
                sft_template_sent_words = tokenizer.tokenize(sft_template_sent.lower())

                if sft_template_sent.lower().find('option') != -1 and sft_template_sent.lower().find('***') != -1:
                    sft_template_sent_words = sft_template_sent_words[1:]

                sft_template_sent_words = list(bigrams(sft_template_sent_words))

                ph_rm_sent = re.sub(r'\[.*?\]', '',sft_template_sent)
                ph_rm_words = tokenizer.tokenize(ph_rm_sent.lower())
                if sft_template_sent.lower().find('option') != -1 and sft_template_sent.lower().find('***') != -1:
                    ph_rm_words = ph_rm_words[1:]
                ph_rm_words = list(bigrams(ph_rm_words))

                _, sim_ratio = self.jaccard_similarity(sft_sent_bgs, sft_template_sent_words)
                _, sim_ratio_nr = self.jaccard_similarity(sft_template_sent_words, sft_sent_bgs, rigid=False)
                _, sim_ratio_nr_rev = self.jaccard_similarity(sft_sent_bgs, sft_template_sent_words, rigid=False)

                _, ph_rm_sim_ratio = self.jaccard_similarity(sft_sent_bgs, ph_rm_words)
                _, ph_rm_sim_ratio_nr = self.jaccard_similarity(ph_rm_words, sft_sent_bgs, rigid=False)
                _, ph_rm_sim_ratio_nr_rev = self.jaccard_similarity(sft_sent_bgs, ph_rm_words, rigid=False)

                if curr_check:
                    if sft_template_sent.lower().find('option') != -1 and sft_template_sent.lower().find('***') != -1:
                        if sim_ratio >= 0.30 or (sim_ratio > 0.20 and sim_ratio_nr > 0.30 and sim_ratio_nr_rev > .40) or (len(sft_sent_words) < 50 and sim_ratio > 0.20 and sim_ratio_nr > 0.75 and sim_ratio_nr_rev > .30 )or (len(sft_sent_words) < 25 and sim_ratio > 0.15 and sim_ratio_nr > 0.75 and sim_ratio_nr_rev > .20):
                            if any(x in sft_sent_words for x in currency_list):
                                pass
                            else:
                                continue
                        curr_check = 0

                if not inverse:
                    if sim_ratio >= 0.30:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break
                    
                    if sim_ratio > 0.20 and sim_ratio_nr > 0.30 and sim_ratio_nr_rev > .40:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if len(sft_sent_words) < 50 and sim_ratio > 0.20 and sim_ratio_nr > 0.75 and sim_ratio_nr_rev > .30:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if len(sft_sent_words) < 25 and sim_ratio > 0.15 and sim_ratio_nr > 0.75 and sim_ratio_nr_rev > .20:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if len(sft_sent_words) < 15 and sim_ratio > 0.10 and sim_ratio_nr > 0.75 and sim_ratio_nr_rev > .10:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if len(sft_sent_words) < 10 and sim_ratio_nr > 0.80:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if ph_rm_sim_ratio >= 0.30:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break
                    
                    if ph_rm_sim_ratio > 0.20 and ph_rm_sim_ratio_nr > 0.30 and ph_rm_sim_ratio_nr_rev > .40:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if len(sft_sent_words) < 50 and ph_rm_sim_ratio > 0.20 and ph_rm_sim_ratio_nr > 0.75 and ph_rm_sim_ratio_nr_rev > .30:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if len(sft_sent_words) < 25 and ph_rm_sim_ratio > 0.15 and ph_rm_sim_ratio_nr > 0.75 and ph_rm_sim_ratio_nr_rev > .20:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if len(sft_sent_words) < 15 and sim_ratio > 0.10 and ph_rm_sim_ratio_nr > 0.75 and ph_rm_sim_ratio_nr_rev > .10:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                    if len(sft_sent_words) < 10 and ph_rm_sim_ratio_nr > 0.80:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

                else:

                    if sim_ratio_nr >= 0.49:
                        similarity_mapping[swift_sentences.index(sft_sent)][template_sentences.index(sft_template_sent)] = sim_ratio
                        n = 1
                        break

            if n == 0:
                similarity_mapping[swift_sentences.index(sft_sent)] = 'NA'

        return similarity_mapping

    def jaccard_similarity(self, swift_words, temp_words, rigid = True):

        common_words = set([word for word in swift_words if word in temp_words])

        try:

            if rigid:
                similarity_ratio = len(common_words) / (len(swift_words) + len(temp_words))
            else:
                similarity_ratio = len(common_words) / len(temp_words)
        
        except ZeroDivisionError:
            similarity_ratio = 0

        return common_words, similarity_ratio

path = 'C:\\Users\\1596949\\Documents\\Trade_Use_Case\\Gtee_and_ITL_samples\\FITL\\UK\\'
# input_path = '/TradeUAT/FITL_LCs_UAT/fs/Instabase Drive/UK/samples/input/'

fnames = ['STDNRP.txt', 'TRAD3.txt']
trans_keyword_file = 'transaction_keywords.csv'
header_keyword_file = 'header_keywords.csv'
trans_identifier_file = 'transaction_identifiers.csv'
trans_out_file = 'transactions_output.csv'

parser = swift_parser(path, fnames, header_keyword_file, trans_keyword_file, trans_identifier_file, trans_out_file)
parser.extract_header_values()
parser.extract_transaction_values()
parser.mark_difference()
parser.generate_output()
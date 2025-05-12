# timeを<e>で分割→1週間ごとに分割→actも同様に分割
# 縦に並べて新しいcsvに保存

import csv
from datetime import datetime, timedelta
import pandas as pd 
import numpy as np 
import sys

df_shibu22old_act = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog01_edited.csv')
df_shibu22old_time = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog01_edited.csv')
df_shibu22new_act = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog02_edited.csv')
df_shibu22new_time = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog02_edited.csv')
df_shibu23_act = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog03_edited.csv')
df_shibu23_time = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog03_edited.csv')
df_toyosu_act = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog_toyosu_edited.csv')
df_toyosu_time = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog_toyosu_edited.csv')
df_shibu21_act = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog_shibu21_edited.csv')
df_shibu21_time = pd.read_csv('/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog_shibu21_edited.csv')


def split_line_to_segments(line):
    """
    各行のトークン列を '<b>'／'<e>' で分割し、セグメントリストを返す
    """
    segments = []
    current = []
    for token in line:
        if token == '<b>':
            current = ['<b>']
        elif token == '<e>':
            if current:
                current.append('<e>')
                segments.append(current)
        else:
            current.append(token)
    return segments 

def split_segments_by_week(segments):
    """
    セグメントのリストを先頭日時を基準に7日区切りのチャンクに分割
    """
    if not segments:
        return []
    # 各セグメントの最初の日時を日付に変換
    # print('seg;;;;;', segments)
    dates = [datetime.strptime(seg[1], "%Y-%m-%d %H:%M:%S").date() for seg in segments] 
    chunks = []
    # print('date;;;', dates)
    start = dates[0]
    week_end = start + timedelta(days=7)
    current = []
    for seg, date in zip(segments, dates):
        if date < week_end:
            current.append(seg)
        else:
            chunks.append(current)
            current = [seg]
            start = date
            week_end = start + timedelta(days=7)
    if current:
        chunks.append(current)
    return chunks

def flatten_chunks(chunks):
    """
    チャンク化されたセグメントを '<b>' と '<e>' 付きのフラットな行リストに変換
    """
    rows = []
    for chunk in chunks:
        row = ['<b>']
        for seg in chunk:
            row.extend(seg)
        row.append('<e>')
        rows.append(row)
    return rows


def date_of_time_seg(seg):
    """時間セグメントから日付を抽出"""
    # print('in date of time seg;;;;', seg)
    return datetime.strptime(seg[1], "%Y-%m-%d %H:%M:%S").date() # 最初のセグメントだけidが入ってるので3番目の項から始まる


def group_into_weeks(segments): #, date_extractor):
    """
    7日間ごとにセグメントをグループ化
    """
    if not segments:
        return []
    chunks = []
    current = []
    # start_date = date_of_time_seg(segments[0])
    start_date = datetime.strptime(segments[0][2], "%Y-%m-%d %H:%M:%S").date() # 最初のセグメントだけidが入ってるので3番目の項から始まる ok

    for i, seg in enumerate(segments):
        # print('----i;;;;-----', i)
        # if i == 0:
        #     print('seg;;;;', seg)
        #     d = start_date
        # else:
        #     d = date_of_time_seg(seg)
        #     print('i;;;;', i , 'seg;;;;', seg)
        # print('seg;;;;', seg) 
        d = date_of_time_seg(seg)
    
        if d < start_date + timedelta(days=7):
            current.append(seg)
        else:
            chunks.append(current)
            current = [seg]
            start_date = d
        
    if current:
        chunks.append(current)
    # print('ikkaisyuuryou')  
    # print('chunks;;;;', chunks)
    # sys.exit()
    return chunks

def fill_missing_days(weekly_time_segs, weekly_act_segs):
    """
    欠落日にはプレースホルダーを挿入（time: 日付トークン / act: '955'）
    """
    # date -> (time_seg, act_seg)
    date_map = {}
    for t_seg, a_seg in zip(weekly_time_segs, weekly_act_segs):
        date_map[date_of_time_seg(t_seg)] = (t_seg, a_seg)

    start = date_of_time_seg(weekly_time_segs[0])
    filled_time, filled_act = [], []
    for i in range(7):
        d = start + timedelta(days=i)
        if d in date_map:
            t_seg, a_seg = date_map[d]
        else:
            t_seg = ['<b>', d.strftime("%Y-%m-%d"), '<e>']
            a_seg = ['<b>', "955", '<e>']
        filled_time.append(t_seg)
        filled_act.append(a_seg)
    return filled_time, filled_act

def write_chunks_to_csv(chunks, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # print('chunks;;;;', chunks[0])
        # sys.exit()
        for segs in chunks:
            row = ['<b>'] + [token for seg in segs for token in seg] + ['<e>']
            writer.writerow(row[1:])

def process_files(time_file, act_file, out_time, out_act):
    with open(time_file, newline='') as f:
        time_rows = list(csv.reader(f))[1:]
    with open(act_file, newline='') as f:
        act_rows = list(csv.reader(f))[1:]

    all_time_chunks, all_act_chunks = [], []
    count = 0 
    for t_row, a_row in zip(time_rows, act_rows): # 個人ごと
        
        # if count == 1:
        #     userid = t_row[0]
        #     print('userid;;;;', userid)
        # print('t_row;;;;', t_row) # 先頭はidが残ってる
        # print('a_row;;;;', a_row)
        # sys.exit()
        t_segs = split_line_to_segments(t_row) # idは消えてる
        a_segs = split_line_to_segments(a_row)

        userid = (t_row[0]) # fixed
        # print('userid;;;;', userid)
        # sys.exit()
        # print(type(userid))

        weekly_ts = group_into_weeks(t_segs) # 日毎のb-eのリスト
        # weekly_ts = [[userid] + seg for seg in weekly_ts]
        # print('weekly_ts;;;;', weekly_ts)
        # sys.exit()

        # 元セグメントとactセグメントの対応表
        idx_map = {tuple(seg): idx for idx, seg in enumerate(t_segs)}
        # print('kokomade')

        for week in weekly_ts:
            # 元データのactを対応づけ
            act_week = []
            for seg in week:
                idx = idx_map.get(tuple(seg))
                act_week.append(a_segs[idx] if idx is not None else ["955"])

            # 欠落日の補完
            filled_t, filled_a = fill_missing_days(week, act_week)
            # print('userid;;;;', userid, type(userid))
            # # sys.exit()
            # filled_t = [userid] + filled_t
            # filled_a = [userid] + filled_a
            filled_t[0].insert(0, f'{userid}')
            filled_a[0].insert(0, f'{userid}')
            all_time_chunks.append(filled_t)
            all_act_chunks.append(filled_a)

            # print('filled_t;;;;', filled_t)
        

        # all_time_chunks[0] = [t_row[0]] + all_time_chunks[0]
        # print('trow0;;;;', t_row[0], type(t_row[0]))
        # all_time_chunks[count][0].insert(0, t_row[0])  # ← これが一番安全
        # all_act_chunks[count][0].insert(0, a_row[0])
        # count += 1
        # all_act_chunks = [a_row[0]] + all_act_chunks
        # print('all_time_chunks;;;;', all_time_chunks)
        # print('all_act_chunks;;;;', all_act_chunks)
        # sys.exit()

    write_chunks_to_csv(all_time_chunks, out_time)
    write_chunks_to_csv(all_act_chunks, out_act)

if __name__ == "__main__":
    # 実行例
    type_list = ['01', '02', '03', '_toyosu', '_shibu21']

    for type_ in type_list:
        process_files(#df_shibu22old_time, df_shibu22old_act, 
                    f'/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog{type_}_edited.csv',
                    f'/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog{type_}_edited.csv',
                    f"/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/timelog{type_}_for_input.csv", 
                    f"/Users/matsunagatakahiro/Desktop/jrres/PPcameraTG/gpslog/actlog{type_}_for_input.csv")
        

        '''
def process_files(time_file, act_file, out_time, out_act):
    # CSV読み込み（ヘッダ行をスキップ）
    with open(time_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        time_rows = list(reader)
    with open(act_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        act_rows = list(reader)

    all_time_rows = []
    all_act_rows = []

    for t_row, a_row in zip(time_rows, act_rows):
        t_segs = split_line_to_segments(t_row)
        a_segs = split_line_to_segments(a_row)
        t_chunks = split_segments_by_week(t_segs)
        # 時間データと同じ構造でアクティビティをチャンク化
        a_chunks = []
        idx = 0
        for chunk in t_chunks:
            length = len(chunk)
            a_chunks.append(a_segs[idx:idx+length])
            idx += length
        all_time_rows.extend(flatten_chunks(t_chunks))
        all_act_rows.extend(flatten_chunks(a_chunks))

    # 結果を書き出し(to_csvに相当)
    with open(out_time, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in all_time_rows:
            writer.writerow(row)

    with open(out_act, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in all_act_rows:
            writer.writerow(row)
'''

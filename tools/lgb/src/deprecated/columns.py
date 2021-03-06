#feature column
columns = [
    'user_age'
    , 'user_gender'
    , 'user_city'
    , 'user_class2_id'
    , 'user_class2_weight'
    , 'user_tag_id'
    , 'user_tag_weight'
    , 'user_view_history'
    , 'user_view_rowkey_weight'
    , 'user_view_cid_history'
    , 'user_view_c2id_history'
    , 'ctxt_hour'
    , 'ctxt_week_day'
    , 'ctxt_os'
    , 'ctxt_network'
    , 'video_rowkey'
    , 'video_duration'
    , 'video_algorithm_id'
    , 'video_n_cid'
    , 'video_n_cid2'
    , 'video_tag_id'
    , 'video_cover_score'
    , 'video_clarity'
    , 'video_plays_1'
    , 'video_likes_1'
    , 'video_comments_1'
    , 'video_biu_1'
    , 'video_avg_watch_duration_1'
    , 'video_vv_day1_5'
    , 'video_vv_day1_10'
    , 'video_vv_day1_15'
    , 'video_vv_day1_20'
    , 'video_vv_day1_30'
    , 'video_vv_day1_other'
    , 'video_5_vv_rate_1'
    , 'video_10_vv_rate_1'
    , 'video_15_vv_rate_1'
    , 'video_20_vv_rate_1'
    , 'video_30_vv_rate_1'
    , 'video_other_vv_rate_1'
    , 'video_plays_3'
    , 'video_likes_3'
    , 'video_comments_3'
    , 'video_biu_3'
    , 'video_avg_watch_duration_3'
    , 'video_vv_day3_5'
    , 'video_vv_day3_10'
    , 'video_vv_day3_15'
    , 'video_vv_day3_20'
    , 'video_vv_day3_30'
    , 'video_vv_day3_other'
    , 'video_5_vv_rate_3'
    , 'video_10_vv_rate_3'
    , 'video_15_vv_rate_3'
    , 'video_20_vv_rate_3'
    , 'video_30_vv_rate_3'
    , 'video_other_vv_rate_3'
    , 'video_plays_7'
    , 'video_likes_7'
    , 'video_comments_7'
    , 'video_biu_7'
    , 'video_avg_watch_duration_7'
    , 'video_vv_day7_5'
    , 'video_vv_day7_10'
    , 'video_vv_day7_15'
    , 'video_vv_day7_20'
    , 'video_vv_day7_30'
    , 'video_vv_day7_other'
    , 'video_5_vv_rate_7'
    , 'video_10_vv_rate_7'
    , 'video_15_vv_rate_7'
    , 'video_20_vv_rate_7'
    , 'video_30_vv_rate_7'
    , 'video_other_vv_rate_7'
    , 'label'
]

drop_columns = [
    'video_rowkey'
    , 'user_class2_id'
    , 'user_class2_weight'
    , 'user_tag_id'
    , 'user_tag_weight'
    , 'user_view_history'
    , 'user_view_rowkey_weight'
    , 'user_view_cid_history'
    , 'user_view_c2id_history'
    , 'video_rowkey'
    , 'video_tag_id'
    , 'ctxt_network'

]

category_columns_1 = [
    'user_age'
    , 'user_gender'
    , 'user_city'
    , 'ctxt_hour'
    , 'ctxt_week_day'
    , 'ctxt_os'
    , 'video_algorithm_id'
    , 'video_n_cid'
    , 'video_n_cid2'

]
category_columns_2 = [
     'user_class2_id_1'
    , 'user_class2_id_2'
    , 'user_class2_id_3'
    , 'user_class2_id_4'
    , 'user_class2_id_5'
    , 'user_tag_id_1'
    , 'user_tag_id_2'
    , 'user_tag_id_3'
    , 'user_tag_id_4'
    , 'user_tag_id_5'
    #, 'user_view_cid_history_1'
    #, 'user_view_cid_history_2'
    #, 'user_view_cid_history_3'
    #, 'user_view_cid_history_4'
    #, 'user_view_cid_history_5'
    #, 'user_view_cid2_history_1'
    #, 'user_view_cid2_history_2'
    #, 'user_view_cid2_history_3'
    #, 'user_view_cid2_history_4'
    #, 'user_view_cid2_history_5'
    , 'video_tag_id_1'
    , 'video_tag_id_2'
    , 'video_tag_id_3'
    , 'video_tag_id_4'
    , 'video_tag_id_5'
]

target_label = 'label'

if __name__ == '__main__':
    print(category_columns_1 + category_columns_2)

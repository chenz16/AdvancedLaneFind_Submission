
    '''
    left_fitx_last = left_fit_last[0]*y_eval**2 + left_fit_last[1]*y_eval+ left_fit_last[2]
    right_fitx_last = right_fit_last[0]*y_eval**2 + right_fit_last[1]*y_eval+ right_fit_last[2]
    left_fitx       = left_fit[0]*y_eval**2 + left_fit[1]*y_eval+ left_fit[2]
    right_fitx      = right_fit[0]*y_eval**2 + right_fit[1]*y_eval+ right_fit[2]
    left_change     = np.array(left_fitx_last-left_fitx) < 3.7*0.1/xm_per_pix
    right_change    = np.array(right_fitx_last - right_fitx) < 3.7*0.1/xm_per_pix
    curvarad_check_left = curvarad_check_left & left_change[0] & left_change[1]
    curvarad_check_right = curvarad_check_right & right_change[0] & right_change[1]
    '''

        '''
        curvarad_check  = np.sum(curvarad_curret, axis = 0)/curvarad_curret.shape[0]
        curvarad_check_left = (curvarad_check[0] > 200)
        curvarad_check_right = (curvarad_check[1] > 200)
        '''

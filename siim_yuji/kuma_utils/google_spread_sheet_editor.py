import gspread
from oauth2client.service_account import ServiceAccountCredentials

class GoogleSpreadSheetEditor:
    def __init__(self, title):
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name('kuma_utils/gspread.json', scope)
        gc = gspread.authorize(credentials)
        self.wks = gc.open(title).sheet1
        self.fold_col_map = {0: 10, 1:11, 2:12, 3:13, 4:14}


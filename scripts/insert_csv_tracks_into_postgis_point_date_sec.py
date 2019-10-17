import argparse
import csv
import datetime
import psycopg2
    
def insert_track(self, track):       
        
    table = self.args['postgretable']
    
    sql = "INSERT INTO " + table + "(slice, cam, day, part, subpart, track_id, time, track_class, geom) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,ST_GeomFromEWKT(%s))"
          
    try:
        self._cur.execute(sql,track)
           
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)    


class App(object):

    def __init__(self):

        ap = argparse.ArgumentParser()
        ap.add_argument(
            "-r", "--framerate",
            default = 25)
        ap.add_argument(
            "-s", "--slice", 
            default = "Testdatensatz")
        ap.add_argument(
            "-d", "--day", 
            default = "Testdatensatz")
        ap.add_argument(
            "-p", "--part", 
            default = 1)
        ap.add_argument(
            "-b", "--subpart", 
            default = 1)
        ap.add_argument(
            "-y", "--subpartstarttime", 
            default = 1521027720)
        ap.add_argument(
            "-u", "--postgreuser", 
            default= "hcuadmin")
        ap.add_argument(
            "-w", "--postgrepassword",
            default = None)
        ap.add_argument(
            "-i", "--postgreip",
            default = "localhost")
        ap.add_argument(
            "-x", "--postgreport",
            default = 5432)
        ap.add_argument(
            "-e", "--postgredb",
            default = "gisdata")
        ap.add_argument(
            "-t", "--postgretable",
            default = "tracks_points_sec")
        ap.add_argument(
            "-f", "--track_file_path",
            default = None)

        self.args = vars(ap.parse_args())
        
        self._conn = None
        self._cur = None

        if self.args['postgrepassword'] is not None and self.args['postgretable'] is not None and self.args['track_file_path'] is not None:
            
            self._conn = psycopg2.connect(host=self.args['postgreip'],database=self.args['postgredb'], user=self.args['postgreuser'], password=self.args['postgrepassword'], port=self.args['postgreport'])
            self._conn.set_session(autocommit=True)
            self._cur = self._conn.cursor() 
        
        else:
            exit()


    def run(self):
        
        try:
        
            with open(self.args['track_file_path'], 'r') as csv_file:
    
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    next(csv_reader, None)
                    line_count = 0
                    
                    slicee = self.args['slice']
                    day = self.args['day']
                    part = self.args['part']
                    subpart = self.args['subpart']
                    subpartstarttime = int(self.args['subpartstarttime'])
                    framerate = int(self.args['framerate'])
    
                    for row in csv_reader:
                    
                        cam = row[11]
                        image_id = float(row[0])
                        track_id = float(row[1])
                        track_class = int(float(row[7]))   
                                            
                        x_utm = row[13]                    
                        y_utm = row[14]
                        
                        timestamp = datetime.datetime.fromtimestamp(subpartstarttime) + datetime.timedelta(milliseconds = (image_id / framerate) * 1000)
                                    
                        point = 'SRID=5555;POINT({0} {1})'.format(str(x_utm), str(y_utm))
                        
                        insert_track(self, (slicee, cam, day, part, subpart, track_id, timestamp.strftime('%Y-%m-%d %H:%M:%S'), track_class, point))
                                                    
                        line_count += 1
                        
        finally:
            
            if self._cur is not None: self._cur.close()
            if self._cur is not None: self._conn.close()

if __name__ == '__main__':
    #    import sys

    App().run()

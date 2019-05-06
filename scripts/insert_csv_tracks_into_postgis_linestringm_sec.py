import argparse
import csv
import datetime
import psycopg2
    
def insert_track(self, track):       
        
    table = self.args['postgretable']
    sql = "INSERT INTO " + str(table) + "(slice, cam, day, part, subpart, starttime, endtime, track_time_range, frame_rate, track_class, track_id, geom) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,ST_GeomFromEWKT(%s))"
          
    try:
        self._cur.execute(sql,track)
        self._conn.commit()
           
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
            default = 4)
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
            default = "tracks_linestrings_milli")
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

        with open(self.args['track_file_path'], 'r') as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader, None)               
                
                slicee = self.args['slice']
                day = self.args['day']
                part = self.args['part']
                subpart = self.args['subpart']
                subpartstarttime = int(self.args['subpartstarttime'])
                framerate = int(self.args['framerate'])
                
                former_track_id = None
                first_track_img_num = None
                last_track_img_num = None
                lineStringMString = None
                former_cam = None
                former_timestamp = None  
                
                for row in csv_reader:
                
                    image_id = float(row[0])
                    track_id = float(row[1])
                    cam = row[11]
                    track_class = int(float(row[7]))   
                                        
                    x_utm = row[13]                    
                    y_utm = row[14]
                    
                    if(track_class == 1 or track_class == 2 or track_class == 3 or track_class == 4 or track_class == 6 or track_class == 8):
                    
                        if former_track_id is None:
                            
                            former_track_id = track_id
                            former_track_class = track_class
                            former_cam = cam
                            last_track_img_num = image_id
                            first_track_img_num = image_id
                            timestamp = datetime.datetime.fromtimestamp(subpartstarttime) + datetime.timedelta(milliseconds=((image_id / framerate) * 1000))
                            former_timestamp = timestamp
                            lineStringMString = []
                            lineStringMString.append("SRID=5555;LINESTRINGM(%s %s %s" % (
                                str(x_utm),
                                str(y_utm),
                                int((subpartstarttime + (image_id / framerate)) * 1000)))
                            
                        elif track_id != former_track_id:
                                                  
                                                   
                            lineStringMString.append(")")
                
                            starttime = datetime.datetime.fromtimestamp(subpartstarttime) + datetime.timedelta(milliseconds=((first_track_img_num / framerate) * 1000))                    
                            endtime = datetime.datetime.fromtimestamp(subpartstarttime) + datetime.timedelta(milliseconds=((last_track_img_num / framerate) * 1000))                    
                        
                            track_time_range = "[" + starttime.strftime('%Y-%m-%d %H:%M:%S.%f') + ", " + endtime.strftime('%Y-%m-%d %H:%M:%S.%f') + "]"
                            
                            insert_track(self, (slicee, day, former_cam, part, subpart, starttime.strftime('%Y-%m-%d %H:%M:%S.%f'), endtime.strftime('%Y-%m-%d %H:%M:%S.%f'), track_time_range, framerate, former_track_class, former_track_id, ''.join(lineStringMString)))
                            
                            lineStringMString = []
                            lineStringMString.append("SRID=5555;LINESTRINGM(%s %s %s" % (
                                str(x_utm),
                                str(y_utm),
                                int((subpartstarttime + (image_id / framerate)) * 1000)))
                
                            timestamp = datetime.datetime.fromtimestamp(subpartstarttime) + datetime.timedelta(milliseconds=((image_id / framerate) * 1000))
                            former_timestamp = timestamp
                            former_track_id = track_id
                            first_track_img_num = image_id
                            last_track_img_num = image_id
                            former_track_class = track_class
                            former_cam = cam
                        
                        else:
                            timestamp = datetime.datetime.fromtimestamp(subpartstarttime) + datetime.timedelta(milliseconds=((image_id / framerate) * 1000))                    
                        
                            if former_timestamp + datetime.timedelta(seconds=1) <= timestamp:
                        
                                last_track_img_num = image_id
                                lineStringMString.append(",%s %s %s" % (
                                    str(x_utm),
                                    str(y_utm),
                                    int((subpartstarttime + (image_id / framerate)) * 1000)))    
                                
                                former_timestamp = timestamp                     
       
    def __del__(self):
        
        if self._cur is not None: self._cur.close()
        if self._cur is not None: self._conn.close()

if __name__ == '__main__':
    #    import sys

    App().run()

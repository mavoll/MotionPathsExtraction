import argparse
import csv
import psycopg2
    
def insert_track(self, track):       
        
    sql = "INSERT INTO tracks(track_id, cam_id, slice_id, day, part, subpart, track_class, starttime, endtime, frame_rate, geom) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,ST_GeomFromEWKT(%s))"
          
    try:
        self._cur.execute(sql,track)
        # self._conn.commit()
           
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)    


class App(object):

    def __init__(self):

        ap = argparse.ArgumentParser()
        ap.add_argument(
            "-r", "--framerate",
            default = 25)
        ap.add_argument(
            "-c", "--cam",
            default = None)
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
            "-f", "--track_file_path",
            default = None)

        self.args = vars(ap.parse_args())
        
        self._conn = None
        self._cur = None

        if self.args['postgrepassword'] is not None and self.args['cam'] is not None and self.args['track_file_path'] is not None:
            
            self._conn = psycopg2.connect(host=self.args['postgreip'],database=self.args['postgredb'], user=self.args['postgreuser'], password=self.args['postgrepassword'], port=self.args['postgreport'])
            self._conn.set_session(autocommit=True)
            self._cur = self._conn.cursor() 
        
        else:
            exit()


    def run(self):

        # VALUES  (track_id, 1, 4, 'Testdatensatz', 1, 1, track_class, starttime, endtime, 25, LineStringM),
        
        with open(self.args['track_file_path'], 'r') as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader, None)
                line_count = 0
                
                cam = self.args['cam']
                slicee = self.args['slice']
                day = self.args['day']
                part = self.args['part']
                subpart = self.args['subpart']
                subpartstarttime = self.args['subpartstarttime']
                framerate = self.args['framerate']
                
                former_track_id = None
                first_track_img_num = None
                last_track_img_num = None
                lineStringMString = None
                
                former_timestamp_milli = None

                for row in csv_reader:
                
                    image_id = int(row[0])
                    track_id = int(row[1])
                    track_class = int(float(row[7]))   
                                        
                    x_utm = row[13]                    
                    y_utm = row[14]
                    
                    timestamp = subpartstarttime + ((image_id / framerate)) # * 1000)
            
                    timestamp_milli = timestamp * 1000
                        
                    if former_track_id is None:
                        
                        former_track_id = track_id
                        first_track_img_num = image_id
                        lineStringMString = []
                        lineStringMString.append("SRID=5555;LINESTRINGM(%s %s %s" % (
                            str(x_utm),
                            str(y_utm),
                            str(timestamp)))
            
                        former_timestamp_milli = timestamp_milli
                        
                    elif track_id != former_track_id:
                        
                        former_track_id = track_id
                        first_track_img_num = image_id
                        starttime = subpartstarttime + ((first_track_img_num / framerate)) # * 1000)  
                        endtime = subpartstarttime + (((last_track_img_num - first_track_img_num) / framerate)) # * 1000) 
                        lineStringMString.append(")")
                        
                        insert_track(self, (track_id, cam, slicee, day, part, subpart, track_class, starttime, endtime, framerate, ''.join(lineStringMString)))
                        
                        lineStringMString = []
                        lineStringMString.append("SRID=5555;LINESTRINGM(%s %s %s" % (
                            str(x_utm),
                            str(y_utm),
                            str(timestamp)))
            
                        former_timestamp_milli = timestamp_milli
                    
                    else:
                        
                        if former_timestamp_milli + 1000 <= timestamp_milli:
                            last_track_img_num = image_id
                            lineStringMString.append(",%s %s %s" % (
                                str(x_utm),
                                str(y_utm),
                                str(timestamp)))
                            former_timestamp_milli = timestamp_milli
                            
                    
                    line_count += 1
       
    def __del__(self):
        
        if self._cur is not None: self._cur.close()
        if self._cur is not None: self._conn.close()

if __name__ == '__main__':
    #    import sys

    App().run()

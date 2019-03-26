import numpy as np
import cv2
import colorsys

from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching

import tensorflow as tf

from deep_sort.deep_sort import kalman_filter
from deep_sort.deep_sort import linear_assignment
from deep_sort.deep_sort import iou_matching

class DeepSortTracker:
    
    def __init__(self, deep_sort_feature_model, deep_sort_max_cosine_distance, deep_sort_nn_budget, per_process_gpu_mem_fraction):
        
        metric = nn_matching.NearestNeighborDistanceMetric(
                    "cosine", deep_sort_max_cosine_distance, deep_sort_nn_budget)
        self.tracker = Tracker(metric)
        
        self.encoder = DeepSortTracker.create_box_encoder(deep_sort_feature_model, batch_size=32, per_process_gpu_mem_fraction=0.1)
        
        
    def track(self, vis, cls_boxes, frame_id, deep_sort_min_detection_height, deep_sort_min_confidence, deep_sort_nms_max_overlap):
        
        tracking_boxes = []
        result_boxes = []
        
        deep_sort_boxes = self.generate_deep_sort_detections(vis, cls_boxes, self.encoder)
        
        for row in deep_sort_boxes:
            
            bbox, confidence, class_id, feature = row[0:4], row[4], row[5], row[6:]
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]           
            if bbox[3] < deep_sort_min_detection_height:
                continue
            tracking_boxes.append(DeepSortDetection(bbox, confidence, int(class_id), feature))
               
        
        tracking_boxes = [d for d in tracking_boxes if d.confidence >= deep_sort_min_confidence]
        

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in tracking_boxes])
        scores = np.array([d.confidence for d in tracking_boxes])
        indices = preprocessing.non_max_suppression(
            boxes, deep_sort_nms_max_overlap, scores)
        tracking_boxes = [tracking_boxes[i] for i in indices]
        
        # Update tracker.
        self.tracker.predict()
         
        
        self.tracker.update(tracking_boxes) 
            
        result_boxes = self.get_confirmed_tracks(frame_id)
            
        return result_boxes
        
    def generate_deep_sort_detections(self, vis, cls_boxes, encoder=None):
        
        features = encoder(vis, cls_boxes[:, 0:4].copy())
        deep_sort_boxes = [np.r_[(cls_boxes, feature)] for cls_boxes, feature
                           in zip(cls_boxes, features)]
        
        return deep_sort_boxes

    def create_box_encoder(model_filename, input_name="images",
        output_name="features", batch_size=32, per_process_gpu_mem_fraction=0.1):
        image_encoder = DeepSortImageEncoder(model_filename, input_name, output_name, per_process_gpu_mem_fraction)
        image_shape = image_encoder.image_shape
    
        def encoder(image, boxes):
            image_patches = []
            for box in boxes:
                box = box
                patch = DeepSortTracker.extract_image_patch(image, box, image_shape[:2])
                
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(box))
                    patch = np.random.uniform(
                        0., 255., image_shape).astype(np.uint8)
                image_patches.append(patch)
            image_patches = np.asarray(image_patches)
            return image_encoder(image_patches, batch_size)
    
        return encoder
    
    def extract_image_patch(image, bbox, patch_shape):
        """Extract image patch from bounding box.
    
        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.
    
        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.
    
        """
        bbox = np.array(bbox)
        
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width
    
        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)
    
        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image
    
    def get_confirmed_tracks(self, frame_id):
        
        tracking_boxes = []
        
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            tracking_boxes.append([
                frame_id, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track.class_id])
            
        return tracking_boxes
      
    def draw_trackers(self, im, tracking_boxes, dataset, thresh=0.9):
                
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlwh()
        
            im = self.vis_bbox(
                    im, (bbox[0], bbox[1], bbox[2], bbox[3]), track.track_id)
            
            im = self.vis_track_id(im, (bbox[0], bbox[1] - 2), track.track_id, dataset[track.class_id])            
    
        return im
    
    def vis_track_id(self, img, pos, track_id, class_id, font_scale=0.35):
        """Visualizes the class."""
        
        DeepSortTracker.create_unique_color_uchar(track_id)
        class_str = 'track_' + str(track_id) + ' (' + str(class_id) + ')'
        
        #img = img.astype(np.uint8)
        x0, y0 = int(pos[0]), int(pos[1])
        # Compute text size.
        txt = class_str
        font = cv2.FONT_HERSHEY_SIMPLEX
        ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
        # Place text background.
        back_tl = x0, y0 - int(1.3 * txt_h)
        back_br = x0 + txt_w, y0
        cv2.rectangle(img, back_tl, back_br, DeepSortTracker.create_unique_color_uchar(track_id), -1)
        # Show text.
        txt_tl = x0, y0 - int(0.3 * txt_h)
        cv2.putText(img, txt, txt_tl, font, font_scale, (218, 227, 218), lineType=cv2.LINE_AA)
        return img
    
    
    def vis_bbox(self, img, bbox, track_id, thick=1):
        """Visualizes a bounding box."""
        #img = img.astype(np.uint8)
        (x0, y0, w, h) = bbox
        x1, y1 = int(x0 + w), int(y0 + h)
        x0, y0 = int(x0), int(y0)
        cv2.rectangle(img, (x0, y0), (x1, y1), DeepSortTracker.create_unique_color_uchar(track_id), thickness=thick)
        return img
    
    def create_unique_color_uchar(tag, hue_step=0.41):
        """Create a unique RGB color code for a given track id (tag).
    
        The color code is generated in HSV color space by moving along the
        hue angle and gradually changing the saturation.
    
        Parameters
        ----------
        tag : int
            The unique target identifying tag.
        hue_step : float
            Difference between two neighboring color codes in HSV space (more
            specifically, the distance in hue channel).
    
        Returns
        -------
        (int, int, int)
            RGB color code in range [0, 255]
    
        """
        r, g, b = DeepSortTracker.create_unique_color_float(tag, hue_step)
        return int(255*r), int(255*g), int(255*b)
    
    def create_unique_color_float(tag, hue_step=0.41):
        """Create a unique RGB color code for a given track id (tag).
    
        The color code is generated in HSV color space by moving along the
        hue angle and gradually changing the saturation.
    
        Parameters
        ----------
        tag : int
            The unique target identifying tag.
        hue_step : float
            Difference between two neighboring color codes in HSV space (more
            specifically, the distance in hue channel).
    
        Returns
        -------
        (float, float, float)
            RGB color code in range [0, 1]
    
        """
        h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
        r, g, b = colorsys.hsv_to_rgb(h, 1., v)
        return r, g, b
 

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, detection.class_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1


class TrackState(object):
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track(object):
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, class_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.class_id = class_id

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
class DeepSortDetection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, class_id, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.class_id = int(class_id)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class DeepSortImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features",
                 per_process_gpu_mem_fraction=0.1):
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_mem_fraction      
        self.session = tf.Session(config=config)
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        self._run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out
    
    def _run_in_batches(self, f, data_dict, out, batch_size):
        data_len = len(out)
        num_batches = int(data_len / batch_size)
    
        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
            out[s:e] = f(batch_data_dict)
        if e < len(out):
            batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
            out[e:] = f(batch_data_dict)
    
    
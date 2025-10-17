;; Auto-generated. Do not edit!


(when (boundp 'flow_publisher::FlowDataMsg)
  (if (not (find-package "FLOW_PUBLISHER"))
    (make-package "FLOW_PUBLISHER"))
  (shadow 'FlowDataMsg (find-package "FLOW_PUBLISHER")))
(unless (find-package "FLOW_PUBLISHER::FLOWDATAMSG")
  (make-package "FLOW_PUBLISHER::FLOWDATAMSG"))

(in-package "ROS")
;;//! \htmlinclude FlowDataMsg.msg.html


(defclass flow_publisher::FlowDataMsg
  :super ros::object
  :slots (_flow_x_integral _flow_y_integral _ground_distance ))

(defmethod flow_publisher::FlowDataMsg
  (:init
   (&key
    ((:flow_x_integral __flow_x_integral) 0)
    ((:flow_y_integral __flow_y_integral) 0)
    ((:ground_distance __ground_distance) 0.0)
    )
   (send-super :init)
   (setq _flow_x_integral (round __flow_x_integral))
   (setq _flow_y_integral (round __flow_y_integral))
   (setq _ground_distance (float __ground_distance))
   self)
  (:flow_x_integral
   (&optional __flow_x_integral)
   (if __flow_x_integral (setq _flow_x_integral __flow_x_integral)) _flow_x_integral)
  (:flow_y_integral
   (&optional __flow_y_integral)
   (if __flow_y_integral (setq _flow_y_integral __flow_y_integral)) _flow_y_integral)
  (:ground_distance
   (&optional __ground_distance)
   (if __ground_distance (setq _ground_distance __ground_distance)) _ground_distance)
  (:serialization-length
   ()
   (+
    ;; int16 _flow_x_integral
    2
    ;; int16 _flow_y_integral
    2
    ;; float32 _ground_distance
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int16 _flow_x_integral
       (write-word _flow_x_integral s)
     ;; int16 _flow_y_integral
       (write-word _flow_y_integral s)
     ;; float32 _ground_distance
       (sys::poke _ground_distance (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int16 _flow_x_integral
     (setq _flow_x_integral (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int16 _flow_y_integral
     (setq _flow_y_integral (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; float32 _ground_distance
     (setq _ground_distance (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get flow_publisher::FlowDataMsg :md5sum-) "66f15ac7ac0281db583d79d134e42e32")
(setf (get flow_publisher::FlowDataMsg :datatype-) "flow_publisher/FlowDataMsg")
(setf (get flow_publisher::FlowDataMsg :definition-)
      "int16 flow_x_integral
int16 flow_y_integral
float32 ground_distance

")



(provide :flow_publisher/FlowDataMsg "66f15ac7ac0281db583d79d134e42e32")



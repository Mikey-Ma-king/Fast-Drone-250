; Auto-generated. Do not edit!


(cl:in-package flow_publisher-msg)


;//! \htmlinclude FlowDataMsg.msg.html

(cl:defclass <FlowDataMsg> (roslisp-msg-protocol:ros-message)
  ((flow_x_integral
    :reader flow_x_integral
    :initarg :flow_x_integral
    :type cl:fixnum
    :initform 0)
   (flow_y_integral
    :reader flow_y_integral
    :initarg :flow_y_integral
    :type cl:fixnum
    :initform 0)
   (ground_distance
    :reader ground_distance
    :initarg :ground_distance
    :type cl:float
    :initform 0.0))
)

(cl:defclass FlowDataMsg (<FlowDataMsg>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <FlowDataMsg>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'FlowDataMsg)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name flow_publisher-msg:<FlowDataMsg> is deprecated: use flow_publisher-msg:FlowDataMsg instead.")))

(cl:ensure-generic-function 'flow_x_integral-val :lambda-list '(m))
(cl:defmethod flow_x_integral-val ((m <FlowDataMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader flow_publisher-msg:flow_x_integral-val is deprecated.  Use flow_publisher-msg:flow_x_integral instead.")
  (flow_x_integral m))

(cl:ensure-generic-function 'flow_y_integral-val :lambda-list '(m))
(cl:defmethod flow_y_integral-val ((m <FlowDataMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader flow_publisher-msg:flow_y_integral-val is deprecated.  Use flow_publisher-msg:flow_y_integral instead.")
  (flow_y_integral m))

(cl:ensure-generic-function 'ground_distance-val :lambda-list '(m))
(cl:defmethod ground_distance-val ((m <FlowDataMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader flow_publisher-msg:ground_distance-val is deprecated.  Use flow_publisher-msg:ground_distance instead.")
  (ground_distance m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <FlowDataMsg>) ostream)
  "Serializes a message object of type '<FlowDataMsg>"
  (cl:let* ((signed (cl:slot-value msg 'flow_x_integral)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'flow_y_integral)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ground_distance))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <FlowDataMsg>) istream)
  "Deserializes a message object of type '<FlowDataMsg>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'flow_x_integral) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'flow_y_integral) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ground_distance) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<FlowDataMsg>)))
  "Returns string type for a message object of type '<FlowDataMsg>"
  "flow_publisher/FlowDataMsg")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FlowDataMsg)))
  "Returns string type for a message object of type 'FlowDataMsg"
  "flow_publisher/FlowDataMsg")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<FlowDataMsg>)))
  "Returns md5sum for a message object of type '<FlowDataMsg>"
  "66f15ac7ac0281db583d79d134e42e32")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'FlowDataMsg)))
  "Returns md5sum for a message object of type 'FlowDataMsg"
  "66f15ac7ac0281db583d79d134e42e32")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<FlowDataMsg>)))
  "Returns full string definition for message of type '<FlowDataMsg>"
  (cl:format cl:nil "int16 flow_x_integral~%int16 flow_y_integral~%float32 ground_distance~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'FlowDataMsg)))
  "Returns full string definition for message of type 'FlowDataMsg"
  (cl:format cl:nil "int16 flow_x_integral~%int16 flow_y_integral~%float32 ground_distance~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <FlowDataMsg>))
  (cl:+ 0
     2
     2
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <FlowDataMsg>))
  "Converts a ROS message object to a list"
  (cl:list 'FlowDataMsg
    (cl:cons ':flow_x_integral (flow_x_integral msg))
    (cl:cons ':flow_y_integral (flow_y_integral msg))
    (cl:cons ':ground_distance (ground_distance msg))
))

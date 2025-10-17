// Auto-generated. Do not edit!

// (in-package flow_publisher.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class FlowDataMsg {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.flow_x_integral = null;
      this.flow_y_integral = null;
      this.ground_distance = null;
    }
    else {
      if (initObj.hasOwnProperty('flow_x_integral')) {
        this.flow_x_integral = initObj.flow_x_integral
      }
      else {
        this.flow_x_integral = 0;
      }
      if (initObj.hasOwnProperty('flow_y_integral')) {
        this.flow_y_integral = initObj.flow_y_integral
      }
      else {
        this.flow_y_integral = 0;
      }
      if (initObj.hasOwnProperty('ground_distance')) {
        this.ground_distance = initObj.ground_distance
      }
      else {
        this.ground_distance = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type FlowDataMsg
    // Serialize message field [flow_x_integral]
    bufferOffset = _serializer.int16(obj.flow_x_integral, buffer, bufferOffset);
    // Serialize message field [flow_y_integral]
    bufferOffset = _serializer.int16(obj.flow_y_integral, buffer, bufferOffset);
    // Serialize message field [ground_distance]
    bufferOffset = _serializer.float32(obj.ground_distance, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type FlowDataMsg
    let len;
    let data = new FlowDataMsg(null);
    // Deserialize message field [flow_x_integral]
    data.flow_x_integral = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [flow_y_integral]
    data.flow_y_integral = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [ground_distance]
    data.ground_distance = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'flow_publisher/FlowDataMsg';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '66f15ac7ac0281db583d79d134e42e32';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int16 flow_x_integral
    int16 flow_y_integral
    float32 ground_distance
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new FlowDataMsg(null);
    if (msg.flow_x_integral !== undefined) {
      resolved.flow_x_integral = msg.flow_x_integral;
    }
    else {
      resolved.flow_x_integral = 0
    }

    if (msg.flow_y_integral !== undefined) {
      resolved.flow_y_integral = msg.flow_y_integral;
    }
    else {
      resolved.flow_y_integral = 0
    }

    if (msg.ground_distance !== undefined) {
      resolved.ground_distance = msg.ground_distance;
    }
    else {
      resolved.ground_distance = 0.0
    }

    return resolved;
    }
};

module.exports = FlowDataMsg;

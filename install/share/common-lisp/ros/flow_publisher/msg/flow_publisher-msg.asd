
(cl:in-package :asdf)

(defsystem "flow_publisher-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "FlowDataMsg" :depends-on ("_package_FlowDataMsg"))
    (:file "_package_FlowDataMsg" :depends-on ("_package"))
  ))
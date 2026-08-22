# DARPA TC E3

## Download

DARPA sub datasets are compressed and stored in a Google Drive folder. We can download them using the following command:

```bash
# pip install gdown
gdown "https://drive.google.com/uc?id=1GG1aUnPjjzzdbxznVTN8X6oVfA-K4oIV" -O ta1-trace-e3-official-1.json.tar.gz
```

_(You specify the file name of the subdataset file you wanna download.)_

Then we do unzip the content into the subdataset folder:

```bash
tar -xzvf ta1-trace-e3-official-1.json.tar.gz -C trace
```

The `trace` folder will contain the subdataset files as shown below in json format:

```txt title="Trace subdataset content"
total 20G
3.2G Apr 19  2018 ta1-trace-e3-official-1.json
3.0G Apr 19  2018 ta1-trace-e3-official-1.json.1
3.0G Apr 19  2018 ta1-trace-e3-official-1.json.2
3.0G Apr 19  2018 ta1-trace-e3-official-1.json.3
3.0G Apr 19  2018 ta1-trace-e3-official-1.json.4
3.0G Apr 19  2018 ta1-trace-e3-official-1.json.5
1.3G Apr 19  2018 ta1-trace-e3-official-1.json.6
```

## Format

Each json file contains a list of dataums following the [CDM](https://drive.google.com/file/d/1bhkU9My_MkuSl_MymRPofKslkdkcZ-Yw/view?usp=drive_link) _(Common Data Model 18v)_ schema.

```json title="Snapshot of Data in a JSON File"
{"datum":{"com.bbn.tc.schema.avro.cdm18.TimeMarker":{"tsNanos":1523627786654000000}},"CDMVersion":"18","source":"SOURCE_LINUX_
SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.StartMarker":{"sessionNumber":3}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRA
CE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.Host":{"uuid":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","hostName":"ta1-trace","hostIdent
ifiers":[{"idType":"serial number","idValue":"fd7d09c4c02369d10350bbb95aa925cc"}],"osDetails":"#192-Ubuntu","hostType":"HOST_D
ESKTOP","interfaces":[{"name":"em2","macAddress":"10:98:36:af:cf:14","ipAddresses":["fe80:0000:0000:0000:1298:36ff:feaf:cf14"]
},{"name":"virbr0","macAddress":"e2:ac:15:92:60:95","ipAddresses":["192.168.122.1"]},{"name":"em2.128","macAddress":"10:98:36:
af:cf:14","ipAddresses":["fe80:0000:0000:0000:1298:36ff:feaf:cf14","128.55.12.118"]},{"name":"em2.10","macAddress":"10:98:36:a
f:cf:14","ipAddresses":["fe80:0000:0000:0000:1298:36ff:feaf:cf14","10.0.6.68"]},{"name":"lo","macAddress":"","ipAddresses":["0
000:0000:0000:0000:0000:0000:0000:0001","127.0.0.1"]}]}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.SrcSinkObject":{"uuid":"39E846F3-D581-6BBB-4CE1-E7E43D356616","baseObject":{"hostId":"
E621F964-5A66-0F89-30E0-67ADB2A5EC28","permission":null,"epoch":{"int":0},"properties":{"map":{"pid":"412"}}},"type":"SRCSINK_
UNKNOWN","fileDescriptor":{"int":8}}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.Principal":{"uuid":"29895546-B124-1BEC-E91C-C9107B81C616","type":"PRINCIPAL_LOCAL","ho
stId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","userId":"0","username":null,"groupIds":["0","0"],"properties":{"map":{"euid":"0"
}}}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject":{"uuid":"753366C8-7B00-E70F-1E95-2102227BD6E1","type":"SUBJECT_PROCESS","cid"
:412,"parentSubject":null,"hostId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","localPrincipal":"29895546-B124-1BEC-E91C-C9107B81C6
16","startTimestampNanos":0,"unitId":{"int":0},"iteration":null,"count":null,"cmdLine":null,"privilegeLevel":null,"importedLib
raries":null,"exportedLibraries":null,"properties":{"map":{"name":"systemd-udevd","seen time":"1523627788.470","ppid":"1"}}}},
"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.Event":{"uuid":"E87FB82D-6375-C469-6974-AACF2B7F1700","sequence":{"long":36},"type":"E
VENT_RECVMSG","threadId":{"int":412},"hostId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","subject":{"com.bbn.tc.schema.avro.cdm18.
UUID":"753366C8-7B00-E70F-1E95-2102227BD6E1"},"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"39E846F3-D581-6BBB-4CE1-
E7E43D356616"},"predicateObjectPath":null,"predicateObject2":null,"predicateObject2Path":null,"timestampNanos":152362778847000
0000,"name":null,"parameters":null,"location":null,"size":{"long":8},"programPoint":null,"properties":{"map":{}}}},"CDMVersion
":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
```

The purpose of the CDM is to provide a common data model for the Transparent Computing (TC) datasets in a way it does capture the entities and interactions out of the source text logs. Each datum is a JSON object that contains a `datum` field and a `CDMVersion` field. The `datum` field contains a `com.bbn.tc.schema.avro.cdm18.TimeMarker` object that contains a `tsNanos` field that represents the timestamp of the event in nanoseconds. The `CDMVersion` field contains the version of the CDM used. The `type` field contains the type of the datum _(for an entity or event/interaction)_. Here is the `.avdl` file that defines the schema of the [CDM v.18](https://github.com/raytheonbbn/tc-ta3-serialization-schema/blob/master/avro/CDM18.avdl).

If we take a look on the CDM v.18 schema, we can identify the following entities/events types:

```java title="Snapshot of the CDM v.18 schema for Entities / Events Types"
/**
    * HostType enumerates the host roles or device types
    */
enum HostType {
    HOST_MOBILE,
    HOST_SERVER,
    HOST_DESKTOP
}

/**
    * SubjectType enumerates the types of execution contexts supported.
    *
    * SUBJECT_PROCESS,    process
    * SUBJECT_THREAD,     thread within a process
    * SUBJECT_UNIT        so far we only know of TRACE BEEP using this
    */
enum SubjectType {
    SUBJECT_PROCESS,
    SUBJECT_THREAD,
    SUBJECT_UNIT,
    SUBJECT_BASIC_BLOCK
}

/**
* PrincipalType identifies the type of user: either local to the
* host, or remote users/systems.
*/
enum PrincipalType {
    PRINCIPAL_LOCAL,            // a principal local on the host
    PRINCIPAL_REMOTE            // a remote principal
}

enum EventType {

    /* Object -> Subject */
    EVENT_ACCEPT,                 // accept a connection on an object

    /* Object1 -> Object2 */
    EVENT_ADD_OBJECT_ATTRIBUTE,   // add attribute to an object that was incomplete at time of publish

    /* Subject -> Object*/
    EVENT_BIND,                   // bind to a network endpoint object

    EVENT_BLIND,                  // blind event for a black boxes that are not instrumented

    /* non-directional */
    EVENT_BOOT,                   // indicates the sytem has booted

    /* Subject -> Object */
    EVENT_CHANGE_PRINCIPAL,       // change the principal associated with the process

    ...

    /* Subject -> Object */
    EVENT_WRITE_SOCKET_PARAMS     // write parameters of a socket
}
```

As we mentioned before, the **Event** item in the CDM schema represents the **Interaction** between **Entities** _(Subjects/Objects)_. This Event items requires a `subject` property _(`UUID` of the Source Subject)_, an optional `predicateObject` and `predicateObject2` that represent the Target Subject/Object -(`UUID`s of the Destination Subject/Object).

```java title="Snapshot of the Event schema"
record Event {

    /** A universally unique identifier for the event */
    UUID uuid;

    /**
     * A logical sequence number for ordering events relative to
     * each other within a subject's execution context
     *
     * This attribute is only optional for inferred events, such
     * as an object's attribute change that was observed without
     * an explicit event or system call.
     */
    union {null, long} sequence = null;

    /** The type of the event */
    EventType type;

    /**
     * UUID of Subject that generated this event.  The subject is
     * required for all events, except the
     * EVENT_ADD_OBJECT_ATTRIBUTE and EVENT_FLOWS_TO event.
     */
    union {null, UUID} subject = null;

    /**
     * UUID of Object/Subject this event acts on. For events that
     * have two arguments, this attribute contains the first
     * argument (following the argument order in the underlying
     * system call). This attribute is optional because it may not
     * be relevant for some events.
     */
    union {null, UUID} predicateObject = null;

    /**
     * Optional UUID of Object/Subject for events that take two
     * arguments (e.g., link, rename, etc). This attribute
     * contains the second argument (following the argument order
     * in the underlying system call).
     */
    union {null, UUID} predicateObject2 = null;
    
    ...

}
```

From a Graph perspective, we have:
- **Nodes**: Subjects/Objects.
- **Edges**: Events.


## Preprocessing

The CDM format provide a lot of ready to use information. We already have the **Nodes** and **Edges** with their _types_ and _attributes_.

For instance, if we want a simple graph representation out of the CDM format, we can define the following mapping:
- **Nodes**: (UUID, type)
- **Edges**: (SrcUUID, DstUUID, type)


## Reference

- Go to [GitHub](https://github.com/darpa-i2o/Transparent-Computing)
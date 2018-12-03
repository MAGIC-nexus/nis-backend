"""
Serialize a model once it has been built
Before solving and/or after solving. Clone operations executed. Scales not performed.

It could be something like:

Metadata:
{
{,
Parameters:
[
 {
 },
],
Categories:
[
 {
 },
],
Observers:
[
 {
 },
],
InterfaceTypes:
[
 {
 },
],
InterfaceTypeConverts: (if conversions not done already)
[
 {
 },
],
Processors:
[
 {
  Interfaces:
  [
   {...,
    Observations:
    [
     {
     },
    ]
   }
  ]
 }
],
Relationships:
[
 {
  Origin: {name, uuid}
  Destination: {name, uuid}
  Type
 }
]
"""
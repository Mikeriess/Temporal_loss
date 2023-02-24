Hospital Billing - Event Log
===============================================================================
DOI: doi:10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741
Author: Felix Mannhardt (Eindhoven University of Technology)
E-Mail: f.mannhardt@tue.nl

Description
===============================================================================
The 'Hospital Billing' event log was obtained from the financial modules of the
ERP system of a regional hospital. The event log contains events that are 
related to the billing of medical services that have been provided by the 
hospital. Each trace of the event log records the activities executed to bill 
a package of medical services that were bundled together. The event log does 
not contain information about the actual medical services provided by the 
hospital.

The 100,000 traces in the event log are a random sample of process instances 
that were recorded over three years. Several attributes such as the 'state' of
the process, the 'caseType', the underlying 'diagnosis' etc. are included in 
the event log. Events and attribute values have been anonymized. The time 
stamps of events have been randomized for this purpose, but the time between
events within a trace has not been altered.

More information about the event log can be found in the related publications.

Please cite as:
Mannhardt, F (Felix) (2017) Hospital Billing - Event Log. 
Eindhoven University of Technology. Dataset. 
https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741
===============================================================================

Attributes
===============================================================================
Name			| Description
-------------------------------------------------------------------------------
actOrange		| A flag that is used in connected with services that 
			| may not be covered by the standard health insurance.
actRed			| A flag that is used in connected with services that 
			| may not be covered by the standard health insurance. 
blocked			| flag that is used when the billing may not proceeed 
			| (i. e., is blocked).
caseType		| A code for the type of the billing package, which 
			| may influence it's handling.
closeCode		| There may be several reasons to close a billing 
			| package, this attribute stores the code used.
diagnosis		| A code for the diagnosis used in the billing package.
flagA			| An anonymized flag.
flagB			| An anonymized flag.
flagC			| An anonymized flag.
flagD			| An anonymized flag.
isCancelled		| A flag that indicates whether the billing package 
			| was eventually cancelled.
isClosed		| A flag that indicates whether the billing package 
			| was eventually closed.
msgCode			| The code returned.
msgCount		| The number of messages returned.
msgType			| The type of messages returned.
speciality		| A code for the medical speciality involved.
state			| Stores the current state of the billing package.
version			| A code for the version of the rules 
			| governing the process.

# The Brain Recording Analysis and Visualization Online (BRAVO) Platform - Server-side Rendering Version

Percept Analysis Platform is a Python-based Web Application designed to process and analyze Medtronic Percept Neurostimulator data. 

Currently a demo site is setup at https://bravo.jcagle.solutions/ to demonstrate its capabilitiy. This Demo site is hosted on a cloud server for the purpose of demonstration. Registration is currently disabled on the server, and the demo is provided via a pre-generated Demo account. You can access the Demo account by clicking on "View Demo Account" button on the main toolbar. The demo account has limited functionalities, and

The demo website is only hosted to provide demonstration purpose, and you should host your own server on your own secure network or local network by following the Installation Process at the Documentation Page (https://bravo.jcagle.solutions/documentation/installation). Although a "Clinician" version with PHI is available, we do not recommend using them unless you receive approval from your institution to keep PHI on your server. 

The tool is completely open sourced, with template from Creative Tim's Argon Dashboard HTML template (https://www.creative-tim.com/product/argon-dashboard) and rewritten by us in Django's template engine language. 

## Version 2 Migration

Version 2 of the BRAVO Platform is currently in development (https://github.com/Fixel-Institute/BRAVO), and it is currently the recommended version to start if you are new to the platform. Changelogs can be found at the new documentation page (https://bravo-documentation.jcagle.solutions/ChangeLogs/v2.0.0).

The primary reason that Version 2 is not being actively maintained from this repository is because we have changed the server organization and utilized a static One-Page approach to the frontend to allow easier manipulation by users. A migration guide is available at (https://bravo-documentation.jcagle.solutions/Tutorials/MigrationGuide) for migrating from BRAVO_SSR to the new BRAVO repository. 

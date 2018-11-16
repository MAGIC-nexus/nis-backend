# GIT client hooks

From the client, you can execute any combination of shell commands before or after specific GIT commands are executed. On each project you can have different scripts which are located at folder `.git/hooks/`. Check this folder for samples.

For more information see:

* https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
* https://es.atlassian.com/git/tutorials/git-hooks

## Hook _pre-commit_

With this hook we can execute shell commands before a `git commit`. If the script exits with 0 (success) the commit will be performed and if it exits with 1 (error) it won't.

In this implementation: 
* _we want to perform the commit only after the units tests have been passed_
* _it also stops the commit if we are working directly with the develop or master branches_  

## Hook _pre-push_

With this hook we can execute shell commands before a `git push`. If the script exits with 0 (success) the push will be performed and if it exits with 1 (error) it won't.

In this implementation: 
* _if we are working with the branches develop or master we want to perform the push  only after the integration tests have been passed_

## Installation instructions

0. Copy the sample files to the project folder `.git/hooks/`
0. Remove the `.sample.sh` suffix
0. Make files executables with command `chmod +x <filename>`
0. Change the PYTHON variable in all files to use your Phyton environment

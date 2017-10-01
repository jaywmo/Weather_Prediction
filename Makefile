# make master

# Define variables here.
branch = tests
message = "Created .py scripts. Ready for V1.1"

master:
	git add .
	git commit -m $(message)
	git push origin

branch:
	git add .
	git commit -m $(message)
	git push origin	$(branch)

update:
	git pull origin master

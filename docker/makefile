#@meta {desc: 'build and deployment for docker projects', date: '2024-03-31'}


## Build system
#
#
# type of project
PROJ_TYPE=		docker
DOCKER_IMG_NAME=	calamr
DOCKER_USER=		plandes
DOCKER_BUILD_OBJS=	prepimage
DOCKER_UP_DEPS +=	mkvol
ADD_CLEAN_ALL +=	$(MNT_DIR)

## Project
#
MNT_DIR=		mnt
PKG_DIR=		$(MTARG)/packages
APP_DIR=		$(MTARG)/app


## Includes
#
include ../zenbuild/main.mk


# compile and install packages
$(PKG_DIR):
			mkdir -p $(PKG_DIR)
			for i in .. ../../propbankdb ; do \
				make -C $$i clean package ; \
				cp $$i/target/pydist/*.whl $(PKG_DIR) ; \
			done

# create the app dir and configuration
$(APP_DIR):
			mkdir -p $(APP_DIR)
			cp -r ../corpus $(APP_DIR)
			cp ../src/config/dot-calamrrc $(APP_DIR)/app.config
			echo "parse_model = spring" >> $(APP_DIR)/app.config
			sed -i 's/^root_dir = .*/root_dir = .\/data/' $(APP_DIR)/app.config

# create docker image artifacts
.PHONY:			prepimage
prepimage:		$(PKG_DIR) $(APP_DIR)

# create local volumes when using volumes (uncomment docker-compose.yml)
.PHONY:			mkvol
mkvol:
			mkdir -p $(MNT_DIR)/data
			mkdir -p $(MNT_DIR)/model
			mkdir -p $(MNT_DIR)/results

# execute a calamr command in the docker image
.PHONY:			exec
exec:
			$(DOCKER_CMD) exec -it calamr su - devusr --command \
				"calamr $(ARGS)"

# align the liu et al. example and put the results in mnt/results
.PHONY:			align
align:
			make ARGS="aligncorp liu-example" exec
			@echo "see results in mnt/results"

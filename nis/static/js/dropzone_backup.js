(function($){
    'use strict';

    $.fn.dropzone = function(settings){

        var $me = this;

        var options = $.extend({
            width:                  300,                            //width of the div
            height:                 300,                            //height of the div
            progressBarWidth:       300,                            //width of the progress bars
            url:                    '',                             //url for the ajax to post
            filesName:              'files',                        //name for the form submit
            margin:                 0,                              //margin added if needed
            border:                 '2px dashed #ccc',              //border property
            background:             '',
            textColor:              '#ccc',                         //text color
            textAlign:              'center',                       //css style for text-align
            text:                   'Drop files here to upload',    //text inside the div
            uploadMode:             'single',                       //upload all files at once or upload single files, options: all or single
            progressContainer:      '',                             //progress selector if null one will be created

            dropzoneWraper:         'nniicc-dropzoneParent',        //wrap the dropzone div with custom class
            files:                  [],                             //Access to the files that are droped
            maxFileSize:            '10MB',                         //max file size ['bytes', 'KB', 'MB', 'GB', 'TB']
            allowedFileTypes:       '*',                            //allowed files to be uploaded seperated by ',' jpg,png,gif
            clickToUpload:          true,                           //click on dropzone to select files old way
            showTimer:              false,                           //show time that has elapsed from the start of the upload,
            removeComplete:         true,                           //delete complete progress bars when adding new files
            preview:                false,                          //if enabled it will load the pictured directly to the html
            params:                 {},                             //object of additional params

            //functions
            load:                   null,                           //callback when the div is loaded
            progress:               null,                           //callback for the files procent
            uploadDone:             null,                           //callback for the file upload finished
            success:                null,                           //callback for a file uploaded
            error:                  null,                           //callback for any error
            previewDone:            null,                           //callback for the preview is rendered
        }, settings);

        var xhrDone = {};
        var timers = {};
        var timerStartDate = {};
        var uploadIndex = 0;

        if(typeof $.ui == "undefined"){
            jQuery.getScript('https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js', function(){
                $('<link/>', {
                   rel: 'stylesheet',
                   type: 'text/css',
                   href: 'https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.4/themes/smoothness/jquery-ui.css'
                }).appendTo('head');
                init();
            });
        }else{
            init();
        }

        function updateParams(a){
            options.params = a;
        }

        function init(){
            $me.css({
                width: options.width,
                height: options.height,
                border: options.border,
                background: options.background,
                color: options.textColor,
                'text-align': options.textAlign,
                'box-align': 'center',
                'box-pack': 'center'
            });

            $me.hover(function() {
                $(this).css("cursor", "pointer");
            }, function() {
                $(this).css("cursor", "default");
            });

            $me.html(options.text);

            $me.wrap('<div class="'+options.dropzoneWraper+'"></div>');
            $("." + options.dropzoneWraper).css('margin', options.margin);
            if(options.progressContainer === ''){
                options.progressContainer = "."+options.dropzoneWraper;
            }

            if(typeof $me.attr('src') !== 'undefined'){
                var src = $me.attr('src');
                $me.attr('src', '');
                var clone = $me.clone();
                $me.css({
                    'z-index': 200,
                    position: 'absolute'
                }).html('').parent().css('position', 'relative');
                clone.appendTo($me.parent());
                clone.replaceWith('<img id="previewImg" src="'+src+'" />');
                $("#previewImg").css({
                    width: options.width,
                    height: options.height,
                    border: options.border,
                    background: options.background,
                    color: options.textColor,
                    'text-align': options.textAlign,
                    'box-align': 'center',
                    'box-pack': 'center'
                });
            }

            if(options.clickToUpload){
                $("." + options.dropzoneWraper).append('<form></form>');
                var onlyOne = options.preview;
                var multile = "";
                if(!onlyOne) multile = "multiple";
                $("."+options.dropzoneWraper).find('form')
                .append('<input type="file" name="'+options.filesName+'" ' + multile + '/>').hide().
                bind('change', function(event) {
                    $(this).trigger('submit');
                }).on('submit', function(event){
                    event.preventDefault();
                    upload(event.target[0].files);
                    var input = $(this).find('input');

                    //input.wrap('<form>').closest('form').get(0).reset();
                    input.unwrap().hide();
                });
            }

            $me.bind({
                dragover: function(e){
                    e.preventDefault();
                    e.stopPropagation();
                    $me.css({
                        color: '#000',
                        'border-color': '#000'
                    });
                },
                dragleave: function(e){
                    e.preventDefault();
                    e.stopPropagation();
                    dragLeave($me);
                },
                drop: function(e){
                    e.preventDefault();
                    dragLeave($me);
                    if(!options.preview){
                        if(options.url === '') alert('Upload targer not found!! please set it with \'url\' attribute');
                        else
                            upload(e.originalEvent.dataTransfer.files);
                    }else{
                        upload(e.originalEvent.dataTransfer.files);
                    }
                },
                click: function(e){
                    if(options.clickToUpload){
                        var el;
                        var form;
                        if(!options.preview){
                            if(options.url === '') alert('Upload targer not found!! please set it with \'url\' attribute');
                            else{
                                el = $("." + options.dropzoneWraper).find('input');
                                if(el.parent().prop('tagName') !== 'FORM'){
                                    form = $("<form></form>");
                                    form.bind('change', function(){
                                        $(this).trigger('submit');
                                    }).on('submit', function(event){
                                        event.preventDefault();
                                        upload(event.target[0].files);
                                        var input = $(this).find('input');

                                        //input.wrap('<form>').closest('form').get(0).reset();
                                        input.unwrap().hide();
                                    });
                                    el.wrap(form);
                                }
                                el.trigger('click');
                            }
                        }else{
                            el = $("." + options.dropzoneWraper).find('input');
                            if(el.parent().prop('tagName') !== 'FORM'){
                                form = $("<form></form>");
                                form.bind('change', function(){
                                    $(this).trigger('submit');
                                }).on('submit', function(event){
                                    event.preventDefault();
                                    upload(event.target[0].files);
                                    var input = $(this).find('input');

                                    //input.wrap('<form>').closest('form').get(0).reset();
                                    input.unwrap().hide();
                                });
                                el.wrap(form);
                            }
                            el.trigger('click');
                        }
                    }
                }
            });


            if(typeof options.load == "function") options.load($me);

            function dragLeave(me){
                var borderColor = options.textColor;
                var borderCheck = options.border.split(" ");
                if(borderCheck.length == 3) borderColor = borderCheck[2];
                $me.css({
                    color: options.textColor,
                    'border-color': borderColor
                });
            }

            function upload(files){
                if(options.preview){
                    if(!checkFileType(files[0])){
                        if(typeof options.error == "function"){
                            options.error($me, "fileNotAllowed", "File is not allowerd to upload! You can only upload the following files ("+options.allowedFileTypes+")");
                        }else
                            alert("File is not allowerd to upload! You can only upload the following files ("+options.allowedFileTypes+")");
                        return;
                    }
                    if(!checkFileSize(files[0])) {
                        if(typeof options.error == "function"){
                            options.error($me, "fileToBig", 'File to big ('+formatBytes(files[0].size)+')! Max file size is ('+options.maxFileSize+')');
                        }else
                            alert('File to big ('+formatBytes(files[0].size)+')! Max file size is ('+options.maxFileSize+')');
                        return;
                    }
                    var reader = new FileReader();
                    $me.css({
                        'z-index': 200,
                        position: 'absolute'
                    }).html('').parent().css('position', 'relative');
                    var clone = $me.clone();
                    clone.appendTo($me.parent());
                    clone.replaceWith('<img id="previewImg" />');
                    $("#previewImg").css({
                        width: options.width,
                        height: options.height,
                        border: options.border,
                        background: options.background,
                        color: options.textColor,
                        'text-align': options.textAlign,
                        'box-align': 'center',
                        'box-pack': 'center'
                    });
                    reader.onload = function(e){
                        $("#previewImg").attr('src', e.target.result).show();
                        if(typeof options.previewDone == "function") options.previewDone($me);
                    };
                    reader.readAsDataURL(files[0]);
                }else{
                    if(files){
                        options.files = files;
                        if(options.removeComplete){
                            var $removeEls = $(".progress-bar:not(.active)").parents('.extra-progress-wrapper');
                            $removeEls.each(function(index, el) {
                                el.remove();
                            });
                        }
                        var i, formData, xhr;
                        if(options.uploadMode == 'all'){
                            timerStartDate[0] = $.now();

                            formData = new FormData();
                            xhr = new XMLHttpRequest();

                            for (i = 0; i < files.length; i++) {
                                formData.append(options.filesName + '[]', files[i]);
                            }
                            if(Object.keys(options.params).length > 0){
                                for(var key in options.params){
                                    formData.append(key, options.params[key]);
                                }
                            }
                            addProgressBar(0);
                            bindXHR(xhr, 0);


                            xhr.open('post', options.url);
                            xhr.setRequestHeader('Cache-Control', 'no-cache');
                            xhr.send(formData);
                            $(".progress").show();
                        }else if(options.uploadMode == 'single'){
                            for (i = 0; i < files.length; i++) {
                                timerStartDate[uploadIndex] = $.now();

                                formData = new FormData();
                                xhr = new XMLHttpRequest();

                                if(!checkFileType(files[i])){
                                    addWrongFileField(i, uploadIndex);
                                    uploadIndex++;
                                    continue;
                                }
                                if(!checkFileSize(files[i])) {
                                    addFileToBigField(i, uploadIndex);
                                    uploadIndex++;
                                    continue;
                                }
                                formData.append(options.filesName + '[]', files[i]);
                                if(Object.keys(options.params).length > 0){
                                    for(var key in options.params){
                                        formData.append(key, options.params[key]);
                                    }
                                }

                                addProgressBar(i, uploadIndex);
                                bindXHR(xhr, i, uploadIndex);

                                xhr.open('post', options.url);
                                xhr.setRequestHeader('Cache-Control', 'no-cache');
                                xhr.send(formData);
                                $(".progress").show();
                                uploadIndex++;
                            }
                        }
                    }
                    showTooltip();
                }
            }
        }

        function startTimer(i){
            timers[i] = window.setInterval(function(){
                var $el = $(".upload-timer-" + i);

                var diff = $.now() - timerStartDate[i];

                var sec = diff / 1000;
                var min = 0;
                if(sec >= 60){
                    min = Math.round(sec / 60);
                    sec = sec % 60;
                }

                $el.text(min + ":" + pad(sec.toFixed(2), 5));

            }, 10);
        }

        function pad (str, max) {
            str = str.toString();
            return str.length < max ? pad("0" + str, max) : str;
        }

        function bindXHR(xhr, i, index){
            $(xhr.upload).bind({
                progress: function(e){
                    if(e.originalEvent.lengthComputable){
                        var percent = e.originalEvent.loaded / e.originalEvent.total * 100;
                        if(typeof options.progress == "function") options.progress(percent, index);
                        else{
                            //var fileName = file.name.trunc(15);
                            $(".progress-"+index).children().css("width", percent+"%").html(percent.toFixed(0)+"%");
                        }
                    }
                },
                loadstart: function(e, a, b, c){
                    startTimer(index);
                }
            });

            xhrDone[index] = false;

            $(xhr).bind({
                readystatechange: function(){
                    if(this.readyState == 4 && this.status == 200){
                        changeXhrDoneStatus(index);
                        $(".progress.progress-"+index).children().removeClass('active');
                        if(typeof options.success  == "function") options.success(this, index);
                    }
                }
            });

            var interval = setInterval(function(){
                if(Object.keys(xhrDone).length > 0){
                    var allOk = {};

                    for(var indexT in xhrDone){
                        if(xhrDone[indexT] === true) allOk[indexT] = true;
                    }

                    if(Object.keys(xhrDone).length == Object.keys(allOk).length){
                        clearInterval(interval);
                        xhrDone = {};
                        if(typeof options.uploadDone == "function") options.uploadDone($me);
                    }
                }
            }, 500);
        }

        function changeXhrDoneStatus(i){
            xhrDone[i] = true;
            clearInterval(timers[i]);
        }

        function addProgressBar(i, index){
            $(options.progressContainer)
                .append('<div class="progress progress-'+index+'"></div>')
                .css({'margin': options.margin});
            $(".progress-"+index).css({
                width: options.progressBarWidth,
                margin: '20px 0 0 0',
            }).append('<div class="progress-bar progress-bar-info progress-bar-striped active"></div>').hide();
            $(".progress-" + index).wrap('<div class="extra-progress-wrapper"></div>');
            $(".progress-" + index).parent().append('<span title="'+options.files[i].name+'">'+options.files[i].name.trunc(20)+'</span>').css("width", options.progressBarWidth);
            if(options.showTimer){
                $(".progress-" + index).parent().append('<span style="float:right" class="upload-timer-'+index+'">0</span>');
            }
        }

        function addFileToBigField(i, index){
            $(options.progressContainer)
                .append('<div class="progress error-progress-'+index+'"></div>')
                .css('margin', options.margin);
            var file = options.files[i];
            var fileName = file.name.trunc(25);
            $(".error-progress-"+index).css({
                width: options.progressBarWidth,
                margin: '20px 0 0 0'
            }).append('<div class="progress-bar progress-bar-danger progress-bar-striped" style="width:100%">File to big ('+formatBytes(file.size)+')</div>');
            $(".error-progress-" + index).wrap('<div class="extra-progress-wrapper"></div>').css("width", options.progressBarWidth);
            $(".error-progress-" + index).parent().append('<span title="'+options.files[i].name+'">'+fileName+'</span>');
        }

        function addWrongFileField(i, index){
            $(options.progressContainer)
                .append('<div class="progress error-progress-'+index+'"></div>')
                .css('margin', options.margin);
            var file = options.files[i];
            var fileName = file.name.trunc(25);
            var extension = file.name.substr(file.name.lastIndexOf('.') + 1);
            $(".error-progress-"+index).css({
                width: options.progressBarWidth,
                margin: '20px 0 0 0'
            }).append('<div class="progress-bar progress-bar-danger progress-bar-striped" style="width:100%">File type ('+extension+') is not allowed</div>');
            $(".error-progress-" + index).wrap('<div class="extra-progress-wrapper"></div>').css("width", options.progressBarWidth);
            $(".error-progress-" + index).parent().append('<span title="'+options.files[i].name+'">'+fileName+'</span>');
        }

        function showTooltip(){
            $("span").tooltip({
                open: function(event, ui){
                    ui.tooltip.css("max-width", '100%');
                }
            });
        }

        function checkFileType(file){
            if (!file.type && file.size%4096 === 0) return false;
            if(options.allowedFileTypes == '*') return true;
            var extension = file.name.substr(file.name.lastIndexOf('.') + 1).toLowerCase();

            var allowedTypes = options.allowedFileTypes.replace(' ', '').split(",");
            var allowedTypesLower = [];
            for (var i = allowedTypes.length - 1; i >= 0; i--) {
                allowedTypesLower.push(allowedTypes[i].toLowerCase());
            }

            if($.inArray(extension, allowedTypesLower) != -1) return true;

            return false;
        }

        function checkFileSize(file){
            var sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

            var sizeType = options.maxFileSize.match(/[a-zA-Z]+/g)[0];
            var sizeValue = options.maxFileSize.match(/\d+/)[0];
            var sizeIndex = $.inArray(sizeType, sizes);


            if(sizeIndex != -1){
                var fileSize = formatBytes(file.size);
                var fileSizeType = fileSize.match(/[a-zA-Z]+/g)[0];
                var fileSizeValue = fileSize.match(/\d+/)[0];
                var fileSizeIndex;

                if(sizeType == fileSizeType){
                    fileSizeIndex = $.inArray(fileSizeType, sizes);
                    if(parseInt(fileSizeValue) * (Math.pow(1024, fileSizeIndex)) > file.size){
                        return true;
                    }
                }else{
                    fileSizeIndex = $.inArray(fileSizeType, sizes);
                    if(fileSizeIndex > -1){
                        if((parseInt(fileSizeValue) * (Math.pow(1024, fileSizeIndex))) < (parseInt(sizeValue) * (Math.pow(1024, sizeIndex)))){
                            return true;
                        }
                    }
                }
            }else{
                alert("Incorect max file size definition!! ("+sizes.join(',')+")");
            }

            return false;

        }
        function formatBytes(bytes,decimals) {
           var sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
           if (bytes === 0) return '0 Byte';
           var i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
           return Math.round(bytes / Math.pow(1024, i), 2) + sizes[i];
        }
        String.prototype.trunc = String.prototype.trunc || function(n){
              return this.length>n ? this.substr(0,n-1)+'&hellip;' : this;
        };
        $.fn.dropzone = function(options) {
            if (options === 'updateParams') {
                return updateParams.apply(this, Array.prototype.splice.call(arguments, 1));
            }
        };

        return $me;
    };

})(jQuery);

from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from pydrive.files import GoogleDriveFile, FileNotUploadedError, ApiRequestError
import exceptions as ex
import re
import six
from apiclient import errors
import traceback
from apiclient import http
import time
from datetime import datetime
import sys
import os
from os import path
import csv

from sys import stderr as cerr
from multiprocessing import Pool
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

class GoogleClient(object):
    
    
    def __init__(self,logger,meta_path):
        
        self.drive = None
        self.current_directory_object = None
        self.rootID = None
        self.dir = None
        self.logger = logger
        self.google_meta_path = meta_path
        self.init()
         
    
    def init(self):        
        self.drive = self.login()
        self.current_directory_object = self.get_file_by_ID('root')  # cmd-like movement in google drive structure
        self.rootID = self.current_directory_id  # remember root ID
        self.logger.info("Current Directory Id:" + str(self.current_directory_id))
        self.dir = '/'
        self.logger.info("Current Directory :" + str(self.current_directory_object))
        self.service = self.drive.auth.service
        self.row_list = []
        self.pool = ThreadPool(int(10))
         
    def login(self):
        """ function for login """
        self.auth = GoogleAuth()
        self.auth.LocalWebserverAuth()
        self.auth.Authorize()
        self.drive = GoogleDrive(self.auth)
        return self.drive 

    def get_file_by_ID(self, id):
        """Gets the GoogleDriveFile for a file
        Parameters
        ----------
        id : str
            Google UUID
        Returns
        -------
        item : GoogleDriveFile
            object for the file
        """
        metadata = {'id': id}
        item = GoogleDriveFile(self.drive.auth, metadata, uploaded = True)
        try:
            item.FetchMetadata()
        except (FileNotUploadedError, ApiRequestError):
            print('File does not exist')
            return None
        return item
    
    @property
    def current_directory_id(self):
        """ID of current directory
        Returns
        -------
        id : str
            Google UUID
        """
        if self.current_directory_object.metadata['id'] == self.rootID:
            return 'root'
        else:
            return self.current_directory_object.metadata['id']
        
    def get_folder_id(self, folder_name):
        file_list = self.drive.ListFile({'q': "'%s' in parents and trashed=false" % 'root'}).GetList()
        for f in file_list:
            if f['mimeType']=='application/vnd.google-apps.folder': # if folder
                if folder_name == f['title']:
                    print("Folder Name: " + folder_name + "; "  + "Folder Id: " + str(f['id']))
            return f['id']
        
        raise Exception('Folder Not Found' + str("Folder Name:" ) + folder_name)
    
    def rebuild_current_path(self):
        """After a cd, rebuilds the current working directory string
        """
        if self.current_directory_id == 'root':
            self.dir = '/'
            return

        self.dir = ''
        this_directory = self.current_directory_object
        self.dir = '/' + this_directory['title'] + self.dir
        while (not this_directory['parents'][0]['isRoot']):
            this_directory = self.get_file_by_ID(self.current_directory_object['parents'][0]['id'])
            self.dir = '/' + this_directory['title'] + self.dir
        self.dir = str(self.dir)

    def list_objects(self, names_only=False, trashed=False):
        """Gets a list of all files in current directory

        Parameters
        ----------
        names_only : bool
            Returns names only?
        trashed : bool
            Look in trash folder instead?

        Returns
        -------
        : list<str> | list<GoogleDriveFile>
            List of files
        """
        if names_only:
            files = self.drive.ListFile({'q': "'{}' in parents and trashed={}".format(self.current_directory_id, str(trashed).lower())}).GetList()
            out = []
            for f in files:
                out.append(f['title'])
            return out
        return self.drive.ListFile({'q': "'{}' in parents and trashed={}".format(self.current_directory_id, str(trashed).lower())}).GetList()
        
    def cd(self, directory=None, make_if_not_exist=False, isID=False):
        """Changes directory

        Parameters
        ----------
        directory : str
            path
        make_if_not_exist : bool
            make the directory if it doesn't exist?
        isID : bool
            is `directory` a Google UUID?
        Returns
        -------
        : bool
            success of cd operation
        """
        if directory is None or directory == '.':
            print(self.dir)
            return True

        if isID:
            try:
                self.current_directory_object = self.get_file_by_ID(directory)
                self.rebuild_current_path()
                return True
            except ex.FileNotFoundError:
                return False

        directory = re.sub('^\./', '', directory)

        remainder = None
        if '/' in directory:  # not a simple change, so we just process the first one only each turn
            directory = re.sub('/$', '', directory)
            if '/' in directory:  # if it wasn't just a ending slash
                dirs = re.split('/', directory, 1)
                directory = dirs[0]
                remainder = dirs[1]

        if directory == '..':  # move up one
            if self.current_directory_object['parents'][0]['isRoot']:
                self.current_directory_object = self.get_file_by_ID('root')
            else:
                self.current_directory_object = self.get_file_by_ID(self.current_directory_object['parents'][0]['id'])
        elif directory in ['/', '']:  # change to root
            self.current_directory_object = self.get_file_by_ID('root')
        else:  # move down one
            directories = self.drive.ListFile({'q': 'title="{}" and "{}" in parents and mimeType = \'application/vnd.google-apps.folder\' and trashed = false'.format(directory, self.current_directory_id)}).GetList()
            if len(directories) < 1:
                if not make_if_not_exist:
                    print('No such directory: {}'.format(directory))
                    return False
                else:
                    directories.append(self.get_file_by_ID(self.mkdir(directory)))
            if len(directories) > 1:
                print('For some reason there\'s more than one result')
                for item in directories:
                    print('{}\t{}'.format(item['id'], item['title']))
                return False
            self.current_directory_object = directories[0]

        self.rebuild_current_path()
        # recursively cd to the innermost folder
        if remainder is not None:
            return self.cd(remainder)
        return True
    
    @property
    def pwd(self):
        """Prints current working directory

        Returns
        -------
        None
        """
        print(self.dir)
        return None
    
    def list_directory(self, path):
        if path is not None:
            path = re.sub('^\./', '', path)
        current_directory = self.current_directory_object
        if path is not None and len(path) > 0:
            if not self.cd(path):
                return
        list = self.list_objects(names_only = True)
        if path is not None:
            self.current_directory_object = current_directory
            self.rebuild_current_path()
        return list

    def ls(self, directory=None):
        """Prints contents of a folder

        Parameters
        ----------
        directory : str
            folder to list, if None defaults to pwd

        Returns
        -------
        None
        """
        if directory is not None:
            directory = re.sub('^\./', '', directory)
        current_directory = self.current_directory_object
        if directory is not None and len(directory) > 0:
            if not self.cd(directory):
                return
        list = self.list_objects()
        for item in list:
            print('{} {}'.format('d' if item['mimeType'] == 'application/vnd.google-apps.folder' else ' ', item['title']))
        if directory is not None:
            self.current_directory_object = current_directory
            self.rebuild_current_path()



    def mkdir(self, folder_name):
        """

        Parameters
        ----------
        folder_name : str
            name of folder to make

        Returns
        -------
        id : str
            Google UUID of new directory

        """
        # name formatting
        folder_name = re.sub('^\./', '', folder_name)

        # not current dir
        current_directory = None
        if '/' in folder_name:
            current_directory = self.current_directory_object
            tokens = re.split('/', folder_name)
            for i in range(len(tokens) - 1):
                if not self.cd(tokens[i], True):
                    return None
            folder_name = tokens[-1]

        directories = self.drive.ListFile({'q': 'title="{}" and "{}" in parents and mimeType = \'application/vnd.google-apps.folder\' and trashed=false'.format(folder_name, self.current_directory_id)}).GetList()
        if len(directories) > 0:
            print('Folder already exists')
            return None

        folder = self.drive.CreateFile({'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [{'id': self.current_directory_id}]})
        folder.Upload()

        if current_directory is not None:
            self.current_directory_object = current_directory
            self.rebuild_current_path()
        return folder.metadata['id']

    def move(self, file, destination, sb=None, db=None, o=None):
        """

        Parameters
        ----------
        file : str
            file to move
        destination : str
            destination folder

        Returns
        -------
        : bool
            success of move operation
        """

        # TODO: renaming in mv
        origin = re.match('.*/', file)
        destination = re.match('.*/', destination)
        if origin is destination:
            print('renaming using move is not supported yet. use the .rename method')
            return False
        if origin.group(0) == destination.group(0):
            print('renaming using move is not supported yet. use the .rename method')
            return False

        return self.update_metadata(file, {'parents': [{'id': self.get_ID_by_name(destination)}]})

    def rename(self, original_name, new_name):
        """

        Parameters
        ----------
        original_name : str
            file to rename
        new_name : str
            new name

        Returns
        -------
        : bool
            success of operation
        """

        # TODO: name validation
        if '/' in new_name:
            print('No slashes in name, please')
            return False

        return self.update_metadata(original_name, {'title': new_name})

    def copy(self, original_name, copy_name, sb=None, db=None, o=None):
        """

        Parameters
        ----------
        original_name : str
            name of origin
        copy_name : str
            name/path of copy

        Returns
        -------

        """
        # TODO: directory copying and deep/shallowness
        try:
            original = self.get_file_by_name(original_name)
            self.upload_stream(copy_name, original.content)
            copyID = self.get_ID_by_name(copy_name)
            if 'properties' in original.metadata.keys() and len(original.metadata['properties'].keys()) > 0:
                for key in original.metadata['properties'].keys():
                    if not self.insert_property(copyID, key, original.metadata['properties'][key]):
                        print('Property copy for {} failed'.format(key))
            return True
        except Exception as e:
            print (e.__str__())
            return False

    def delete(self, file_name, recursive=False, delete=False):
        """

        Parameters
        ----------
        file_name : str
            name of file/folder to delete
        recursive : bool
            recursively delete a folder?
        delete : bool
            hard delete files?

        Returns
        -------
        : bool
            success of operation
        """
        try:
            fileID = self.get_ID_by_name(file_name)
            f = self.drive.CreateFile({'id': fileID})
            f.FetchMetadata()
            if f.metadata['mimeType'] == 'application/vnd.google-apps.folder' and not recursive:
                print('Is folder and recursive delete is not selected')
                return False
            if delete:
                f.Delete()
            else:
                f.Trash()  # trashed files still exist so the association count remains the same
            return True
        except ex.FileNotFoundError:
            print('File not found')
            return False

    #### File IO functions

    def upload_file(self, file_name, cloud_name=None, permissions=None):
        """Uploads from file on disk

        Parameters
        ----------
        file_name : str
            path to file on disk to upload
        cloud_name : str
            path/name for cloud file

        Returns
        -------
        : bool
            success of operation
        """
        if cloud_name is None:
            cloud_name = file_name

        if not isinstance(file_name, six.string_types):
            # assume this is a file_name-like object
            self.upload_stream(file_name, cloud_name)

        # cloud_name formatting
        cloud_name = re.sub('^\./', '', cloud_name)

        # not current dir
        current_directory = None
        if '/' in cloud_name:
            current_directory = self.current_directory_object
            tokens = re.split('/', cloud_name)
            for i in range(len(tokens) - 1):
                if not self.cd(tokens[i], True):
                    return False
            cloud_name = tokens[-1]
        
        print("Cloud Name: " + cloud_name)
        print("self.current_directory_id: " + self.current_directory_id)
        # metadata
        metadata = {}
        metadata['title'] = cloud_name
        metadata['parents'] = [{'id': self.current_directory_id}]

        # try to upload the file_name
        newfile = self.drive.CreateFile(metadata)
        newfile.Upload()
        try:
            newfile.SetContentFile(file_name)
            newfile.Upload()
        except Exception as e:
            print('Error uploading file_name:\n{}'.format(e.__str__()))
            newfile.delete()
            return False

        if current_directory is not None:
            self.current_directory_object = current_directory
            self.rebuild_current_path()

        return True
    
    def upload_file_to_parent_folder(self, file_name,  parent_id):
        """Uploads from file on disk

        Parameters
        ----------
        file_name : str
            path to file on disk to upload
        cloud_name : str
            path/name for cloud file

        Returns
        -------
        : bool
            success of operation
        """
        file_path = os.path.basename(file_name) 
        
        print("file_path: " + file_path)
    
        # metadata
        metadata = {}
        metadata['title'] = file_path
        metadata['parents'] = [{'id': parent_id}]

        # try to upload the file_name
        newfile = self.drive.CreateFile(metadata)
        newfile.Upload()
        try:
            newfile.SetContentFile(file_name)
            newfile.Upload()
        except Exception as e:
            
            print('Error uploading file_name: ' + str(file_name))
            newfile.delete()
            return False
        
        self.logger.info("File :" + str(file_name) + "; Successfully uploaded.")
        return True
    
    def list_folder(self, parent_id):
        filelist=[]
        file_list = self.drive.ListFile({'q': "'%s' in parents and trashed=false" % parent_id}).GetList()
        for f in file_list:
            if f['mimeType']=='application/vnd.google-apps.folder': # if folder
                filelist.append({"id":f['id'],"title":f['title']})
            else:
                filelist.append(f['title'])
        return filelist
  
    #### Misc helper functions

    def check_ID_exists(self, id):
        """

        Parameters
        ----------
        id : str
            Google UUID

        Returns
        -------
        : bool
            existence of ID on drive
        """
        metadata = {'id': id}
        item = GoogleDriveFile(self.drive.auth, metadata, uploaded = True)
        try:
            item.FetchMetadata()
            return True
        except (FileNotUploadedError, ApiRequestError):
            return False
        print('This line should not be executed.')

    def check_file_exists(self, file_name, bn):
        """

        Parameters
        ----------
        file_name : str
            path on cloud to check

        Returns
        -------
        : bool
            existence of file
        """
        try:
            self.get_ID_by_name(file_name)
            return True
        except ex.FileNotFoundError:
            return False
        print('This line should not be executed.')



    def get_file_by_name(self, name):
        """Gets the GoogleDriveFile for a file
        Parameters
        ----------
        name : str
            name of file to get

        Returns
        -------
        item : GoogleDriveFile
            object for the file
        """
        return self.get_file_by_ID(self.get_ID_by_name(name))

    def get_ID_by_name(self, file_name):
        """Gets the Google UUID for a file

        Parameters
        ----------
        file_name : str
            name of file to get

        Returns
        -------
        id : str
            Google UUID
        """

        drive_file = re.sub('^\./', '', file_name)

        if file_name == '/':
            return 'root'

        # not current dir
        current_directory = None
        if '/' in drive_file:
            current_directory = self.current_directory_object
            tokens = re.split('/', drive_file)
            if ('' in tokens):
                tokens.remove('')
            for i in range(len(tokens) - 1):
                if not self.cd(tokens[i]):
                    raise ex.FileNotFoundError
            drive_file = tokens[-1]

        items = self.list_objects()
        item = None
        for i in items:
            if i['title'] == drive_file:
                item = i
                break

        if current_directory is not None:
            self.current_directory_object = current_directory
            self.rebuild_current_path()

        if item is not None:
            return item['id']
        else:
            raise ex.FileNotFoundError

    def insert_property(self, id, key, value, visibility='PUBLIC'):
        """Adds a custom property to a file

        Parameters
        ----------
        id : str
            Google UUID of file
        key : str
            name of the custom property
        value : str
            value of the custom property
        visibility : 'PUBLIC'|'PRIVATE'
            visibility of the property

        Returns
        -------
        : bool
            operation success

        """

        if visibility not in ['PUBLIC', 'PRIVATE']:
            raise ValueError('Bad visibility value')

        if key == 'key':
            return self.store_encryption_key(id, value)

        body = {'key': key, 'value': value, visibility: visibility}
        # print('Adding {}: {}'.format(key, value))

        try:
            self.service.properties().insert(fileId = id, body = body).execute()
            return True
        except Exception as e:
            print('Error: {}'.format(e.__str__()))
            return False

    def get_file_properties(self, id):
        """Gets the properties for a file

        Parameters
        ----------
        id : str
            Google UUID of file

        Returns
        -------
        properties : dict
            custom metadata
        """
        try:
            f = self.get_file_by_ID(id)
            if f is None:
                print('File does not exist')
            properties = {}
            f.FetchMetadata()
            if 'properties' in f.metadata.keys():
                for p in f.metadata['properties']:
                    properties[p['key']] = p['value']
            return properties
        except Exception as e:
            print('Error: {}'.format(e.__str__()))
            return None

    def store_encryption_key(self, fileID, key64, chunk_size=96):
        """Stores a base64-encoded encryption key

        Parameters
        ----------
        fileID : str
            file UUID to store this to
        key64 : str
            base-64 encoded key
        chunk_size : int
            size in chars to store the key in (there are limits to property lengths)

        Returns
        -------

        """
        if chunk_size > 114:
            print('Chunk size set to 114 because of limitations of metadata size')
            chunk_size = 114
        nChuncks = len(key64) / chunk_size + (len(key64) % chunk_size > 0)
        self.insert_property(fileID, 'keyChunks', str(nChuncks))
        self.insert_property(fileID, 'key', 'in chunks')
        for i in range(nChuncks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            end = end if end < len(key64) else len(key64)
            if not self.insert_property(fileID, 'keyChunk{}'.format(i), key64[start:end]):
                print('Key upload failed')
                return

    def update_metadata(self, file_name, metadata):
        """Updates the custom properties for a file

        Parameters
        ----------
        file_name : str
            Name of file
        metadata : dict
            new metadata

        Returns
        -------
        : bool
            operation success
        """
        try:
            drive_file = self.service.files().get(fileId = self.get_ID_by_name(file_name)).execute()
            for key in metadata.keys():
                drive_file[key] = metadata[key]
            self.service.files().update(fileId = self.get_ID_by_name(file_name), body = drive_file).execute()
            return True
        except ex.FileNotFoundError:
            print('File not found')
            return False
        except Exception as e:
            print('Update failed')
            print(e.__str__())
            return False

    @property
    def size(self):
        files = self.drive.ListFile({'q': "trashed=false"}).GetList()
        sizes = [f.metadata['size'] for f in files]
        return sum(sizes)
    
    def get_parents(self, file_id):
        try:
            parents = self.service.parents().list(fileId=file_id).execute()
            return parents['items'][0]['id']
        except Exception as e:
            errorMsg = traceback.format_exc()
            self.logger.error(errorMsg)   
            
    def get_full_path(self, file_id):
        
        dir = ''
        parent_drive_file = self.get_file_by_ID(file_id)
        next_id = parent_drive_file['id']
        dir = '/' + parent_drive_file['title']
        
        try:
            while (not parent_drive_file['parents'][0]['isRoot']):
                parents = self.service.parents().list(fileId=next_id).execute()
                this_directory =  parents['items'][0]['id']
                parent_drive_file = self.get_file_by_ID(this_directory)
                dir = '/' + parent_drive_file['title'] + dir
                next_id = parent_drive_file['id']
        except Exception as e:
            errorMsg = traceback.format_exc()
            print errorMsg
            self.logger.error(errorMsg) 
        path = str(dir)
        
        return path
    
    def print_files_in_folder(self, folder_id):
        page_token = None
        while True:
            try:
                param = {}
                if page_token:
                    param['pageToken'] = page_token
                children = self.service.children().list(folderId=folder_id, **param).execute()

                for child in children.get('items', []):
                    drive_file = self.get_file_by_ID(child['id'])
                    title = drive_file['title']
                    parent_id = self.get_parents(child['id'])
                    parent_drive_file = self.get_file_by_ID(parent_id)
                    parent_title = parent_drive_file['title']
                    if drive_file['mimeType']=='application/vnd.google-apps.folder': # if folder
                        print ('Parent Id: ' + parent_id + '; Parent Title: '  + parent_title + '; File Id: '  + child['id'] + '; Title: ' + title  + ' IS FOLDER')
                        self.print_files_in_folder(child['id'])
                    else:
                        print ('Parent Id: ' + parent_id + '; Parent Title: '  + parent_title + '; File Id: '  + child['id'] + '; Title: ' + title  + ' IS NOT FOLDER')
                    #self.get_parents(child['id'])
                page_token = children.get('nextPageToken')
                if not page_token:
                    break
            except errors.HttpError, error:
                print 'An error occurred: %s' % error
            break
        
    
    def generate_folder_content(self, folder_id):
        
        page_token = None
        while True:
            try:
                param = {}
                if page_token:
                    param['pageToken'] = page_token
                children = self.service.children().list(folderId=folder_id, **param).execute()
                try:
                    for child in children.get('items', []):
                        drive_file = self.get_file_by_ID(child['id'])
                        title = drive_file['title']
                        parent_id = self.get_parents(child['id'])
                        parent_drive_file = self.get_file_by_ID(parent_id)
                        parent_title = parent_drive_file['title']
                        file_id = child['id']
                        full_path = self.get_full_path(child['id'])
                        if drive_file['mimeType']=='application/vnd.google-apps.folder': # if folder
                            file_type = 'FOLDER'
                            row = [file_type, parent_id, parent_title, full_path, title, file_id]
                            self.row_list.append(row)
                            print "Folder Content Size: " + str(len(self.row_list))
                            self.logger.info("Folder Content Size: " + str(len(self.row_list)))
                            self.generate_folder_content(child['id'])
                        else:
                            file_type = 'FILE'
                            row = [file_type, parent_id, parent_title, full_path, title, file_id]
                            self.row_list.append(row)
                        print "Folder Content Size: " + str(len(self.row_list))
                        self.logger.info("Folder Content Size: " + str(len(self.row_list)))
                    #self.get_parents(child['id'])
                    page_token = children.get('nextPageToken')
                    if not page_token:
                        break
                except Exception as e:
                    errorMsg = traceback.format_exc()  
                    self.logger.error(errorMsg)
                    print errorMsg
                    
            except errors.HttpError, error:
                errorMsg = str(error)
                self.logger.error(errorMsg)
                raise Exception(str(error))
            break    
        return self.row_list
    
    def generate_metadata_file(self, data, file_path=None):
        self.write_metadata(data, suffix = file_path)
        
    def download_to_file(self, file_id, local_file=None):
        
        try:
            f = self.drive.CreateFile({'id': file_id})
        except errors.HttpError, error:
            print 'An error occurred: %s' % error
        
        f.FetchMetadata() 
           
        if local_file is None:
            local_file = f['title']
            
        print("Local File: " + local_file)
        f.GetContentFile(local_file)
        return True
    
    def get_file_by_name_by_id(self, file_id):
        
        drive_file = self.get_file_by_ID(file_id)
        if drive_file is None:
            errorMsg = 'File does not exists: ' + str(file_id) 
            self.logger.error(errorMsg)
            raise Exception(errorMsg)
        
        return drive_file['title']
            
    def check_folder_exists(self, parent_id, folder_name):
        
        directories = self.drive.ListFile({'q': 'title="{}" and "{}" in parents and mimeType = \'application/vnd.google-apps.folder\' and trashed=false'.format(folder_name, parent_id)}).GetList()
        
        if len(directories) > 0:
            parent_folder_name = self.get_file_by_name_by_id(parent_id)
            msg = 'Folder already exists within parent directory. Parent Folder: ' +  str(parent_folder_name)  + "; Folder: " + str(folder_name) 
            self.logger.error(msg)
            return True
        return False
    
    def get_folder_with_parent_id(self, parent_id, folder_name):
        
        directories = self.drive.ListFile({'q': 'title="{}" and "{}" in parents and mimeType = \'application/vnd.google-apps.folder\' and trashed=false'.format(folder_name, parent_id)}).GetList()
        
        file_id = None
        for item in directories:
            file_id = item['id']
            break
                
        return file_id
                
        
        
    def create_folder(self, parent_id, folder_name):
        """

        Parameters
        ----------
        folder_name : str
            name of folder to make

        Returns
        -------
        id : str
            Google UUID of new directory

        """
        # name formatting
        folder_name = re.sub('^\./', '', folder_name)

        if '/' in folder_name:
            tokens = re.split('/', folder_name)
            for i in range(len(tokens) - 1):
                if not self.cd(tokens[i], True):
                    return None
            folder_name = tokens[-1]
        
        folder_metadata = {
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{'id': parent_id}]
        }
        
        directories = self.drive.ListFile({'q': 'title="{}" and "{}" in parents and mimeType = \'application/vnd.google-apps.folder\' and trashed=false'.format(folder_name, parent_id)}).GetList()
        
        if len(directories) > 0:
            parent_folder_name = self.get_file_by_name_by_id(parent_id)
            errorMsg = 'Folder already exists within parent directory. Parent Folder: ' +  str(parent_folder_name)  + "; Folder: " + str(folder_name) 
            self.logger.error(errorMsg)
            raise Exception(errorMsg)

        folder = self.drive.CreateFile(folder_metadata)
        folder.Upload()
        
        self.logger.info("Created folder :" + str(folder_name)  + " Folder Id: " + str(folder.metadata['id']))
        return folder.metadata['id']
  
    def write_metadata(self, data, suffix=None):
        
        header = ["file_type", "parent_id", "parent_name", "full_path", "file_name" , "file_id" ]
      
        current_date = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        
        if suffix:
            file_name = "google_drive_metadata_" + str(suffix) + "_" + str(current_date) + ".csv"
        else:
            file_name = "google_drive_metadata_" + str(current_date) + ".csv"
       
        self.google_meta_path =  os.path.join(self.google_meta_path, file_name)
        
        with open(self.google_meta_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        
        self.logger.info("Goodle Metadata file written at: [%s]" , self.google_meta_path)
        f.close()
        
    def create_folder_rec(self, folder, parent_folder_id):
        
        self.data = []
        print('recursive ON')
        ids = {}
        firstTime = True
        firstParent = True
        for root, sub, files in os.walk(folder):
            par = os.path.dirname(root)
            
            self.logger.info("Parent: " + par)
            if firstTime:
                print "I am here 1"
                parent_id = parent_folder_id
                folder_name = os.path.basename(os.path.normpath(root))
                parent_root = root[:-1]
                firstTime = False
            else:
                folder_name = os.path.basename(root)
                print "I am here 2"
                parent_root = root
                if par in ids.keys():
                    print "I am here 3"
                    parent_id = ids[par]
            
            for item in ids.keys():
                self.logger.info("Key: " + item)
              
            self.logger.info("Root: " + root)
            self.logger.info("Folder Name: " + folder_name)
            self.logger.info("Parent Id: " + str(parent_id))
            self.logger.info("Parent Root: " + str(parent_root))
            
            folder_id = self.create_folder(parent_id, folder_name)
            
            self.logger.info("Folder Id: " + folder_id)
            
            if firstParent:
                first_parent_id = folder_id
                firstParent = False
           
            ids[parent_root] = folder_id
            
            self.data = self.create_files(root, ids, files)
            
        results = self.pool.map(self.process_row, self.data)
        self.logger.info("Uploaded Folder. Folder: " + str(folder) + "Files Count: "  + str(len(self.data)))
                    
        self.pool.close()
        self.pool.join()
        return first_parent_id
            
            #for f in files:
            #    print(root+'/'+f)
    
            #    file_name_path = root + '/' + f 
            #    print "File Name Path: " + file_name_path
            #    head, tail = os.path.split(file_name_path)
            #    print "Parent Path: " + str(head)
            #    file_parent_id = ids[head]
            #    print "Parent Id: " + str(file_parent_id)
            #    file_name = os.path.basename(file_name_path)
            #    print "File Name:" +  file_name
            #    self.upload_file_to_parent_folder(file_name_path,  file_parent_id)
                
    def create_files(self, root_dir, ids_set, file_set):
        
        for f in file_set:
            self.logger.info(root_dir+'/'+f)
    
            file_name_path = root_dir + '/' + f 
            self.logger.info("File Name Path: " + file_name_path)
            head, tail = os.path.split(file_name_path)
            self.logger.info("Parent Path: " + str(head))
            file_parent_id = ids_set[head]
            self.logger.info("Parent Id: " + str(file_parent_id))
            file_name = os.path.basename(file_name_path)
            self.logger.info("File Name:" +  file_name)
            row = [file_name_path,file_parent_id]
            self.data.append(row)
        return self.data
        
    def process_row(self,row):
        
        file_name_path = row[0]
        file_parent_id = row[1]
        self.logger.info("Uploading file : " + str(file_name_path))
        self.upload_file_to_parent_folder(file_name_path,  file_parent_id)
        
        


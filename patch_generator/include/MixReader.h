#include "YamlReader.h"
#include "TextReader.h"

#define GAB_ID "bilodeauIR"
#define PLSC_ID "stcharles2018-v04"
#define KAIST_ID "kaist"

#ifndef PATCHGENERATOR_MIXREADER_H
#define PATCHGENERATOR_MIXREADER_H

class MixReader : public GtReader {
public:
    MixReader();
    void readFile(const MapLine& mapLine);
    void mergeMaps();
    void clearImagePoints();
private:
    YamlReader yamlReader;
    TextReader textReader;
    //KaistReader kaistReader;
};

#endif //PATCHGENERATOR_MIXREADER_H

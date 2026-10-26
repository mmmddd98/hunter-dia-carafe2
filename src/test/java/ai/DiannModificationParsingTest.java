package test.java.ai;

import main.java.ai.AIGear;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;

public class DiannModificationParsingTest {

    @Test
    public void exportsPeptideTerminalModificationsWithoutResidueIndexUnderflow() throws Exception {
        CParameter.fixMods = "1,479";
        CParameter.varMods = "2,480,31";
        AIGear aiGear = new AIGear();
        aiGear.load_mod_map();

        Method diann = AIGear.class.getDeclaredMethod("get_modified_peptide_diann", String.class, String.class, String.class);
        Method generic = AIGear.class.getDeclaredMethod("get_modified_peptide", String.class, String.class, String.class);
        Method encyclopedia = AIGear.class.getDeclaredMethod("get_modified_peptide_encyclopedia", String.class, String.class, String.class);
        diann.setAccessible(true);
        generic.setAccessible(true);
        encyclopedia.setAccessible(true);

        Assert.assertEquals(diann.invoke(aiGear, "GGGGGGG", "Dimethyl:2H(4)13C(2)@N-term", "0"),
                "_[UniMod:510]GGGGGGG_");
        Assert.assertEquals(diann.invoke(aiGear, "GGGGGGG", "Acetyl@N-term", "0"),
                "_[UniMod:1]GGGGGGG_");

        // The other exporters share the same terminal-position helper. Their
        // exact notation is format-specific, but neither may index aa[-1].
        Assert.assertTrue(generic.invoke(aiGear, "GGGGGGG", "Acetyl@N-term", "0").toString().contains("GGGGGGG"));
        Assert.assertTrue(encyclopedia.invoke(aiGear, "GGGGGGG", "Dimethyl:2H(4)13C(2)@N-term", "0").toString().contains("GGGGGGG"));
    }

    @Test
    public void parsesConfiguredPeptideNTermModificationsWithMultiDigitCarafeIds() throws Exception {
        CParameter.fixMods = "1,479";
        CParameter.varMods = "2,480,31";

        AIGear aiGear = new AIGear();
        aiGear.load_mod_map();
        String result = aiGear.get_modification_diann(
                "(UniMod:1)AAAAAM(UniMod:35)K",
                "AAAAAMK"
        );

        Assert.assertTrue(result.contains("@0[42.010565"));
        Assert.assertTrue(result.contains("Oxidation of M@6["));

        String lysineDimethyl = aiGear.get_modification_diann(
                "AAAAK(UniMod:510)",
                "AAAAK"
        );
        Assert.assertTrue(lysineDimethyl.contains("@5[34.063117"));

        String nTermDimethyl = aiGear.get_modification_diann(
                "(UniMod:510)AAAAK",
                "AAAAK"
        );
        Assert.assertTrue(nTermDimethyl.contains("@0[34.063117"));

        // The PeptDeep representation keeps terminal modifications at site 0.
        // The legacy x-string must not attempt to write them to aa[-1].
        String[] acetylPeptDeep = aiGear.convert_modification(result);
        String[] dimethylPeptDeep = aiGear.convert_modification(nTermDimethyl);
        Assert.assertTrue(acetylPeptDeep[1].startsWith("0"));
        Assert.assertEquals(dimethylPeptDeep[1], "0");
        Assert.assertEquals(acetylPeptDeep[0], "Acetyl@N-term;Oxidation@M");
        Assert.assertTrue(dimethylPeptDeep[0].contains("Dimethyl:2H(4)13C(2)@"));

        // Library generation uses CompOmics' default PTM name rather than the
        // DIA-NN reconstruction alias. Both must resolve to the same
        // AlphaPeptDeep modification.
        String[] libraryAcetylPeptDeep = aiGear.convert_modification(
                "Acetylation of peptide N-term@0[42.010565]"
        );
        Assert.assertEquals(libraryAcetylPeptDeep[0], "Acetyl@N-term");
        Assert.assertEquals(libraryAcetylPeptDeep[1], "0");

        Method legacyX = AIGear.class.getDeclaredMethod("convert_modification", String.class, String.class);
        legacyX.setAccessible(true);
        Assert.assertEquals(legacyX.invoke(aiGear, "AAAAAMK", result), "-AAAAA1K-");
        Assert.assertEquals(legacyX.invoke(aiGear, "AAAAK", nTermDimethyl), "-AAAAK-");
    }
}
